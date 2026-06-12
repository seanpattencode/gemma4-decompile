# LLMs Love Pi: What We Found Decompiling Every Gemma 4 Model

We scanned the raw weight data of all four Gemma 4 models (plus Gemma 3 as baseline) with a custom single-file C decompiler. Standard GGUF tools parse metadata and tensor manifests. Ours goes further: scanning billions of weight bytes for anomalies, frequency distributions, and embedded constants.

The models were released April 2, 2026. We analyzed them the next day.

## The Constants

All four models — across two generations, three architectures (dense, MoE, edge), and a 7x parameter range (4B to 31B) — contain the same five math constants embedded as F32 values in their quantized weight blocks:

| Constant | Gemma 3 (4B) | G4 E4B (8B) | G4 26B MoE | G4 31B Dense | Actual |
|---|---|---|---|---|---|
| pi | 3.14163637 | -3.14163923 | 3.14152789 | 3.14163661 | 3.14159265 |
| e | 2.71837449 | 2.71837711 | -2.71830726 | 2.71836185 | 2.71828183 |
| sqrt(2) | 1.41411960 | 1.41421247 | 1.41425633 | 1.41423702 | 1.41421356 |
| ln(2) | 0.69311345 | 0.69305730 | 0.69314492 | -0.69319916 | 0.69314718 |
| phi | 1.61798859 | -1.61797142 | 1.61794782 | 1.61798239 | 1.61803399 |

Detection rate: exactly 31 instances in every model. Approximation accuracy within 0.01%. Both positive and negative versions appear — the sign depends on whether the constant participates in addition or subtraction in the operation it approximates.

These models independently converged to storing pi, e, sqrt(2), ln(2), and the golden ratio in their weights. They store them approximately, repeatedly, across layers — burning weight capacity on values that could be single exact constants.

## What Else We Found

We ran 11 experiments total. Here's the summary, details below.

| # | Finding | Section |
|---|---|---|
| 1 | Shared KV and bottleneck gating are E4B-only | [Architecture](#1-shared-kv-and-bottleneck-gating-are-e4b-only) |
| 2 | 26B is a 128-expert MoE with dual-path FFN | [MoE](#2-the-26b-is-a-128-expert-moe-with-dual-path-ffn) |
| 3 | Math constants in weights — all models, same rate | [Above](#the-constants) |
| 4 | Nibble distributions fingerprint architecture type | [Distributions](#4-nibble-distributions-reveal-architecture-type) |
| 5 | Entropy and dead weight analysis | [Dead weight](#5-entropy-and-dead-weight) |
| 6 | Per-layer entropy is U-shaped | [Layer entropy](#6-per-layer-entropy-is-u-shaped-e4b) |
| 7 | Sliding window pattern: every 6th layer is global | [Attention](#7-sliding-window-pattern-every-6th-layer-is-global) |
| 8 | Vision encoder: E4B shrunk, others unchanged | [Vision](#8-vision-encoder-e4b-shrunk-others-unchanged) |
| 9 | Layer output scales are a built-in pruning map | [Pruning map](#9-layer-output-scales-googles-built-in-pruning-map) |
| 10 | No super weight — architecture eliminates fragility | [Super weight](#10-no-super-weight-architecture-eliminates-single-point-of-failure) |
| 11 | Selective degradation via weight ablation | [Ablation](#11-selective-degradation-via-weight-ablation) |

## Quick Start

```bash
ollama pull gemma4 && ollama pull gemma4:26b && ollama pull gemma4:31b && ollama pull gemma3:4b
cc -O2 -o decompile decompile.c    # zero dependencies
./run.sh                            # scans all models
```

Pre-generated results in `results/`.

## Models

| | Gemma 3 (4B) | Gemma 4 E4B (8B) | Gemma 4 26B MoE | Gemma 4 31B Dense |
|---|---|---|---|---|
| File size | 3.2 GB | 9.6 GB | 17 GB | 19 GB |
| Parameters | 4.3B | 8.0B | 25.8B (3.8B active) | 31.3B |
| Tensors | 883 | 2131 | 1014 | 1189 |
| Text blocks | 34 | 42 | 30 | 60 |
| Embedding | 2560 | 2560 | 2816 | 5376 |
| FFN | 10240 | 10240 | 2112+experts | 21504 |
| Context | 131K | 131K | 262K | 262K |
| License | Gemma Terms | Apache 2.0 | Apache 2.0 | Apache 2.0 |

---

## Detailed Findings

### 1. Shared KV and Bottleneck Gating Are E4B-Only

The 8B E4B model has two architectural features absent from all other sizes:

**Shared KV layers**: 18 of 42 layers share their key-value cache instead of maintaining independent caches. ~43% fewer KV caches, directly enabling the 131K context window at 8B size.

**Layer bottleneck gating**: Every block has `inp_gate.weight` (2560→256), `proj.weight` (256→2560), and `layer_output_scale.weight` (1 scalar). Compresses each layer's input to 10% of its width, with a learned scalar controlling contribution to the residual stream. Layers can effectively gate themselves to zero — learned pruning built into the forward pass.

Neither 26B nor 31B uses these. They have `shared_kv_layers = 0` and `embedding_length_per_layer_input = 0`. These are compression innovations specific to the edge model.

### 2. The 26B Is a 128-Expert MoE With Dual-Path FFN

- 128 experts per block, 8 active per token
- Router (`ffn_gate_inp.weight`, 2816×128), packed expert weights (`ffn_gate_up_exps.weight`, 2816×1408×128), expert down-projections (`ffn_down_exps.weight`, 704×2816×128), per-expert scales
- Shared FFN path alongside experts with its own norms (`post_ffw_norm_1`, `post_ffw_norm_2`, `pre_ffw_norm_2`)
- Only 3.8B of 25.8B parameters active per token

### 4. Nibble Distributions Reveal Architecture Type

The 4-bit quantized value distribution (nibble frequency) has a distinct shape per architecture:

- **Gemma 3 (4B)** — Bimodal: peaks at 5-6 and 9-10, valley at 7. Less converged training.
- **Gemma 4 E4B (8B)** — Gaussian: smooth bell centered at 7-8 (8.7-8.9%). Best-trained distribution.
- **Gemma 4 26B MoE** — U-shaped: high at 0-1 (9.97%) and 14-15 (8.87%), low in middle. MoE expert routing signature — most expert weights are dormant or fully active.
- **Gemma 4 31B Dense** — Mild bimodal: closer to Gemma 3 than E4B despite being the largest model.

### 5. Entropy and Dead Weight

| | Gemma 3 | G4 E4B | G4 26B MoE | G4 31B Dense |
|---|---|---|---|---|
| Shannon entropy (bits/byte) | 7.936 | 7.888 | 7.756 | 7.911 |
| Zero bytes | 0.71% | 0.39% | 0.68% | 0.51% |
| Near-zero Q4_K scales | 18.2% | 18.2% | **26.0%** | 19.9% |
| Low-entropy 4KB blocks | 3 | 21 | 21 | 0 |
| Zero regions (>256 bytes) | 0 | 2 | 0 | 0 |

The 26B has 26% near-zero scale blocks — intentional MoE sparsity. The 31B has zero low-entropy blocks — every byte carries information.

### 6. Per-Layer Entropy Is U-Shaped (E4B)

```
blk.0:  7.778  ████████████████████████████████████████
blk.10: 7.764  █████
blk.20: 7.763  ██                   ← tightest
blk.30: 7.776  ███████████████████████████████████
blk.41: 7.776  ███████████████████████████████████
```

Middle layers have tightest weight distributions. Early layers handle diverse inputs, late layers diverse outputs, middle layers the most compressed representational space.

### 7. Sliding Window Pattern: Every 6th Layer Is Global

```
SWA:    0,1,2,3,4,  6,7,8,9,10,  12,13,14,15,16, ...  (35 layers)
Global: 5,          11,           17,23,29,35,41        (7 layers)
```

5:1 local:global ratio. The model attends locally most of the time, reaching out to full 131K context at regular intervals.

### 8. Vision Encoder: E4B Shrunk, Others Unchanged

| | Gemma 3 | G4 E4B | G4 26B | G4 31B |
|---|---|---|---|---|
| Vision blocks | 27 | **16** | 27 | 27 |
| Vision dim | 1152 | **768** | 1152 | 1152 |
| QAT calibration | no | **yes** | no | no |

E4B aggressively compressed vision (40% fewer blocks, 33% narrower) with QAT calibration. Only E4B adds audio (12 blocks, 1024 dim). Larger models kept Gemma 3's vision unchanged.

### 9. Layer Output Scales: Google's Built-In Pruning Map

Every E4B block has a `layer_output_scale.weight` — one F32 scalar controlling the layer's contribution:

```
blk.0:  0.061  ██
blk.1:  0.160  █████
blk.2:  0.840  ████████████████████████████
...
blk.23: 0.065  ██                            ← near-zero
...
blk.37: 0.887  ██████████████████████████████ ← highest
...
blk.41: 0.445  ███████████████
```

**Validation — zeroing entire layers:**

| Block | Scale | 2+2= | France? | Status |
|---|---|---|---|---|
| blk.23 | 0.065 | **4** | **Paris** | **Survives** |
| blk.0 | 0.061 | \<pad\> | \<pad\> | Dead |
| blk.1 | 0.160 | \<pad\> | \<pad\> | Dead |
| blk.37 | 0.887 | gibberish | gibberish | Incoherent |

**blk.23 is the only layer that can be fully zeroed (~60MB) while producing correct answers.** The scale predicted it. blk.0 has equally low scale but zeroing kills the model — it processes raw embeddings, the entry point to the residual stream. The scale is a pruning map for non-input layers.

### 10. No Super Weight: Architecture Eliminates Single-Point-of-Failure

The "Super Weight" paper (Apple, [arXiv 2411.07191](https://arxiv.org/abs/2411.07191)) found zeroing a single scalar in Llama-7B's `mlp.down_proj` destroys the model (perplexity 5.68 → 763).

Our replication on Gemma 4 E4B:

| Target | Rows zeroed | 2+2= | France? | Degraded? |
|---|---|---|---|---|
| blk.0.ffn_down row 0 | 1 of 10240 | 4 | Paris | No |
| blk.0.ffn_down max-magnitude row | 1 of 10240 | 4 | Paris | No |
| blk.0.ffn_down ALL rows | 10240 of 10240 | 2 | user | **Yes** |
| blk.1.ffn_down ALL rows | 10240 of 10240 | 4 | Paris | No |
| blk.2.ffn_down ALL rows | 10240 of 10240 | 4 | Paris | No |
| blk.3.ffn_down ALL rows | 10240 of 10240 | 4 | Paris | No |

**Gemma 4 has no super weight.** Zeroing any single row produces zero degradation. Zeroing the entire ffn_down tensor (~21MB) on blocks 1-3 produces zero degradation. The super weight phenomenon is architecture-dependent — shared KV and bottleneck gating distribute importance so broadly that no single parameter is critical.

Replication: [`super_weight_g4.py`](https://github.com/seanpattencode/aicombo/blob/main/super_weight_g4.py)

### 11. Selective Degradation via Weight Ablation

Anthropic identified concept-specific features in Claude 3 Sonnet using sparse autoencoders ([Golden Gate Claude, May 2024](https://www.anthropic.com/news/golden-gate-claude)). We tested whether similar selective effects can be observed through weight ablation on quantized GGUF files.

**Method**: Zero progressively smaller chunks of rows in `ffn_gate.weight`, test 20 addition and 20 multiplication problems, same-number controls, cross-layer checks.

**Results** — zeroing rows 1280-1344 (64 of 2560) in blk.20.ffn_gate.weight:

| Category | Baseline | After ablation | Change |
|---|---|---|---|
| Addition (20 problems) | 18/20 | 11/20 | **-39%** |
| Multiplication (20 problems) | 20/20 | 19/20 | -5% |
| Same-number addition (7+8, etc.) | 4/4 | 3/4 | -25% |
| Same-number multiplication (7×8, etc.) | 4/4 | 4/4 | 0% |

**Controls**: different rows in the same layer = no effect. Same rows in different layers (blk.15, blk.25) = no effect. The degradation is specific to these 64 rows in blk.20.

Zeroing 2.5% of a tensor preferentially degrades addition (~39%) over multiplication (~5%). The effect is real and controlled, though partial — consistent with different computational pathways rather than discrete "concept neurons."

Scripts: [`concept_ablation.py`](https://github.com/seanpattencode/aicombo/blob/main/concept_ablation.py), [`concept_ablation2.py`](https://github.com/seanpattencode/aicombo/blob/main/concept_ablation2.py), [`concept_ablation3.py`](https://github.com/seanpattencode/aicombo/blob/main/concept_ablation3.py)

### 12. DiffusionGemma: Same Backbone, Bidirectional — Training Objective Visible in the Weights

DiffusionGemma 26B-A4B (June 2026) is the Gemma 4 26B MoE backbone with a diffusion head. The decompile confirms the backbone is *identical* (block_count 30, embedding 2816, 128 experts / 8 active, expert_ff 704, ctx 262144, `final_logit_softcapping=30`) and isolates exactly what diffusion training changed:

| Delta | Evidence in dump | Meaning |
|---|---|---|
| Bidirectional attention | `attention.causal = false` | The base is causal; this attends both directions over the canvas — enables parallel denoising |
| Fixed canvas | `diffusion.canvas_length = 256` | Denoises 256 tokens per step as a block |
| Self-conditioning head | new `self_cond_{down,gate,up,pre_norm}` tensors | Feeds its own prior prediction back each step (~18M params) |
| Mask = noise | `mask_token_id = 4` now load-bearing | Diffusion denoises from masked tokens |
| Partial rename | `enc_layer_output_scale` (blk.28/29 still `layer_output_scale`) | Encoder framing; half-done rename + name `Dg_Rc0P1_Patched` ⇒ patched over the AR checkpoint |
| Vision stripped | zero `vision.*` metadata, 692 tensors vs 1014 | This GGUF is text-only despite the card claiming image input |

**The objective is measurable in the raw weight statistics** — same backbone, different generation paradigm:

| | Gemma 4 26B (autoregressive) | DiffusionGemma 26B |
|---|---|---|
| Shannon entropy (bits/byte) | 7.756 | **7.939** |
| Nibble distribution | U-shaped (valley ~4.4% mid, spikes 0-1/14-15) | **flat / near-uniform (~7-8%)** |
| Near-zero Q4_K scales | 26.0% | **21.1%** |
| Math constants (pi/e/√2/ln2/φ) | 31 | 31 (unchanged) |

Bidirectional denoising raises weight entropy and flattens the MoE nibble distribution — it uses expert capacity more uniformly than sparse causal routing, smoothing out the autoregressive model's "dormant-or-saturated expert" signature. The 31 embedded math constants survive *exactly*, confirming they live in the shared backbone, untouched by the new objective.

Result: [`results/diffusiongemma_26b.txt`](results/diffusiongemma_26b.txt). Runner: [`aicombo`](https://github.com/seanpattencode/aicombo) (`diffusiongemma.py` — self-installing llama.cpp diffusion build + Q4_K_M GGUF).

---

## What Standard Tools Miss

| Capability | gguf-dump / gguf-py | decompile.c |
|---|---|---|
| Metadata + tensor listing | Yes | Yes |
| Low-entropy block detection | No | Yes |
| ASCII strings in weights | No | Yes |
| Math constant detection | No | Yes |
| Zero region detection | No | Yes |
| Byte/nibble frequency analysis | No | Yes |
| Shannon entropy | No | Yes |
| Q4_K scale distribution | No | Yes |
| Byte pair frequency | No | Yes |
| ELF/Mach-O decompile | No | Yes |

## The Tool

`decompile.c` is a single-file C decompiler (~600 lines) that handles:
- **ELF x86-64**: instruction length decoder, disassembly, symbol resolution, compilable C output
- **Mach-O ARM64/x86-64**: fat binary handling, segment/section parsing, disassembly
- **GGUF**: metadata parsing, tensor manifest, anomaly scan (low-entropy, strings, math constants, zero regions), frequency analysis (byte/nibble/pair, entropy, Q4_K scale stats)

Build: `cc -O2 -o decompile decompile.c` — zero dependencies beyond libc.

## Reproduce

```bash
git clone https://github.com/seanpattencode/gemma4-decompile
cd gemma4-decompile
ollama pull gemma4 && ollama pull gemma4:26b && ollama pull gemma4:31b && ollama pull gemma3:4b
./run.sh
```

## License

Analysis and tool: MIT. Gemma 4 models: Apache 2.0 (Google).
