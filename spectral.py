#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "gguf"]
# ///
# spectral.py — weight-matrix spectral fingerprint + cross-model diversity, from a GGUF.
# Goes past decompile.c's byte stats: dequantizes each 2D weight and does real linear algebra.
#   one model : ./spectral.py model.gguf       -> per-layer stable-rank, spectral-entropy, alpha; model means
#   two models: ./spectral.py A.gguf B.gguf    -> same-named tensors' singular-subspace diversity (1 - overlap)
# alpha = Hill power-law exponent of the eigenvalue (sigma^2) tail; lower/heavier-tailed ~ better-trained
# (Martin & Mahoney, "Predicting trends ... without test data", Nat.Comm. 2021). Honest caveat: correlational.
import os
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"  # 1 thread = ~50x faster here (no contention)
import sys, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

LAYERS = [0, 6, 12, 18, 24, 29]                    # sample every ~6th block (fast, representative)
KINDS  = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
MAXROW = 2048                                       # subsample output rows: bounds the Gram cost
RANK   = 16                                          # top singular subspace dim for the diversity compare

def want(name):
    if not name.startswith("blk.") or not name.endswith(".weight"): return False
    if "exps" in name or "norm" in name or "inp" in name: return False   # skip experts, norms, router
    p = name.split(".")
    return len(p) >= 3 and p[1].isdigit() and int(p[1]) in LAYERS and any(k in name for k in KINDS)

def mat(t):                                          # dequantize a ReaderTensor -> 2D float32, row-subsampled
    sh = [int(x) for x in t.shape]
    if len(sh) != 2: return None
    w = dequantize(t.data, t.tensor_type).reshape(sh).astype(np.float32)
    if w.shape[0] > MAXROW: w = w[:: w.shape[0] // MAXROW][:MAXROW]
    return w

def spectrum(w):                                     # eigvals of W Wt (= sigma^2) desc, + eigvecs (left sing. vectors)
    ev, U = np.linalg.eigh(w @ w.T)
    return ev[::-1], U[:, ::-1]

def hill_alpha(ev):                                  # power-law exponent of the top eigenvalue tail
    ev = ev[ev > 1e-12]
    k = max(10, len(ev) // 5)
    if k + 1 >= len(ev): k = len(ev) - 2
    if k < 2: return float("nan")
    return 1.0 + k / np.sum(np.log(ev[:k] / ev[k]))

def fingerprint(w):
    ev = np.clip(np.linalg.eigvalsh(w @ w.T)[::-1], 0, None)   # values only (no eigvecs) = faster
    stable = ev.sum() / ev[0]                          # effective rank = ||W||_F^2 / ||W||_2^2
    p = ev / ev.sum()
    return stable, float(-(p * np.log(p + 1e-30)).sum() / np.log(len(p))), hill_alpha(ev)

def load(path):
    return {t.name: t for t in GGUFReader(path).tensors if want(t.name)}

def one(path):
    ts = load(path)
    print(f"\n=== {path}  ({len(ts)} weight matrices sampled) ===")
    print(f"{'tensor':<26}{'shape':<14}{'stable_rank':>12}{'spec_entropy':>14}{'alpha':>8}")
    A = []
    for name in sorted(ts):
        w = mat(ts[name])
        if w is None: continue
        sr, h, a = fingerprint(w)
        if a == a: A.append(a)
        print(f"{name:<26}{str(tuple(w.shape)):<14}{sr:>12.1f}{h:>14.3f}{a:>8.2f}")
    if A: print(f"  -> mean alpha {np.mean(A):.3f}  (lower = heavier-tailed / better-trained, correlational)")

def two(pa, pb):
    A, B = load(pa), load(pb)
    shared = sorted(set(A) & set(B))
    print(f"\n=== diversity: {pa.split('/')[-1]}  vs  {pb.split('/')[-1]}  ({len(shared)} shared tensors) ===")
    print(f"{'tensor':<26}{'subspace_overlap':>18}{'diversity':>11}")
    divs = []
    for name in shared:
        wa, wb = mat(A[name]), mat(B[name])
        if wa is None or wb is None or wa.shape != wb.shape: continue
        ua = spectrum(wa)[1][:, :RANK]; ub = spectrum(wb)[1][:, :RANK]
        cos = np.linalg.svd(ua.T @ ub, compute_uv=False)          # cosines of principal angles
        overlap = float(np.clip(cos, 0, 1).mean()); divs.append(1 - overlap)
        print(f"{name:<26}{overlap:>18.3f}{1-overlap:>11.3f}")
    if divs: print(f"  -> mean diversity {np.mean(divs):.3f}  (0 = identical subspaces, 1 = orthogonal)")

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if len(a) == 1: one(a[0])
    elif len(a) == 2: one(a[0]); one(a[1]); two(a[0], a[1])
    else: print("usage: spectral.py MODEL.gguf [MODEL2.gguf]"); sys.exit(1)
