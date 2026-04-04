# Gemma 4 Decompile: Weight-Level Analysis of Google's Open Model Family

Released April 2, 2026. Analyzed April 3, 2026 — day one.

We ran a custom single-file C decompiler (`decompile.c`) on all four Gemma 4 GGUF models plus Gemma 3 as baseline. The tool parses GGUF metadata and tensor manifests like standard tools, then goes further: scanning raw tensor data for anomalies, frequency distributions, math constants embedded in weights, and dead weight regions. These findings are not available from any standard GGUF analysis tool.

## Quick Start

```bash
# Pull models
ollama pull gemma4           # 9.6 GB (E4B, 8B params)
ollama pull gemma4:26b       # 17 GB  (MoE, 25.8B params, 3.8B active)
ollama pull gemma4:31b       # 19 GB  (Dense, 31.3B params)
ollama pull gemma3:4b        # 3.2 GB (baseline)

# Build and run
cc -O2 -o decompile decompile.c
./run.sh
# Or individually:
./decompile /path/to/model.gguf
```

Pre-generated results are in `results/`.

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

## Key Findings

### 1. Shared KV and Bottleneck Gating Are E4B-Only

The 8B E4B model has two architectural features absent from all other sizes:

**Shared KV layers**: 18 of 42 layers share their key-value cache instead of maintaining independent caches. This is ~43% fewer KV caches, directly enabling the 131K context window at 8B size.

**Layer bottleneck gating**: Every block has `inp_gate.weight` (2560→256), `proj.weight` (256→2560), and `layer_output_scale.weight` (1 scalar). This compresses each layer's input to 10% of its width, with a learned scalar controlling how much the layer contributes to the residual stream. Layers can effectively gate themselves to zero — learned pruning built into the forward pass.

Neither 26B nor 31B uses these tricks. They have `shared_kv_layers = 0` and `embedding_length_per_layer_input = 0`. These are compression innovations specific to the edge model.

### 2. The 26B Is a 128-Expert MoE With Dual-Path FFN

The 26B has the most exotic architecture of any Gemma model:

- 128 experts per block, 8 active per token
- Each block contains: a router (`ffn_gate_inp.weight`, 2816×128), packed expert weights (`ffn_gate_up_exps.weight`, 2816×1408×128), expert down-projections (`ffn_down_exps.weight`, 704×2816×128), per-expert scales
- A shared FFN path runs alongside the expert path with its own norms (`post_ffw_norm_1`, `post_ffw_norm_2`, `pre_ffw_norm_2`)
- Only 3.8B of 25.8B parameters are active per token

### 3. Math Constants in Weights — All Models, Same Rate

All four models contain the same five fundamental math constants embedded as F32 values within quantized weight blocks:

| Constant | Gemma 3 | G4 E4B | G4 26B | G4 31B | Actual |
|---|---|---|---|---|---|
| pi | 3.14163637 | -3.14163923 | 3.14152789 | 3.14163661 | 3.14159265 |
| e | 2.71837449 | 2.71837711 | -2.71830726 | 2.71836185 | 2.71828183 |
| sqrt(2) | 1.41411960 | 1.41421247 | 1.41425633 | 1.41423702 | 1.41421356 |
| ln(2) | 0.69311345 | 0.69305730 | 0.69314492 | -0.69319916 | 0.69314718 |
| phi | 1.61798859 | -1.61797142 | 1.61794782 | 1.61798239 | 1.61803399 |

Detection rate: exactly 31 instances in every model regardless of size (4B to 31B), architecture (dense vs MoE), or generation (Gemma 3 vs 4). Approximation accuracy is within 0.01% — too precise for coincidence across 31 independent occurrences per model.

These constants are stored approximately, repeatedly, across layers — wasting weight capacity. A dedicated constant neuron would store them once, exactly.

### 4. Nibble Distributions Reveal Architecture Type

The 4-bit quantized value distribution (nibble frequency) has a distinct shape per architecture:

**Gemma 3 (4B)** — Bimodal: peaks at nibbles 5-6 (7.8%) and 9-10 (8.1%), valley at 7 (5.4%). Suggests less converged training — weight distribution hasn't fully collapsed to Gaussian.

**Gemma 4 E4B (8B)** — Gaussian: smooth bell centered at 7-8 (8.7-8.9%), symmetric tails. Best-trained weight distribution of all four models.

**Gemma 4 26B MoE** — U-shaped: high at nibbles 0-1 (9.97%) and 14-15 (8.87%), low in the middle (4.4%). This is the MoE expert routing signature — most expert weights are either dormant (near-zero) or fully active, with few in between. This distribution is unique to MoE models.

**Gemma 4 31B Dense** — Mild bimodal: closer to Gemma 3 than E4B, peaks at 6 and 9-10. Despite being the largest model, its weight distribution is less clean than the 8B — possibly undertrained relative to capacity, or the larger model has more diverse feature representations that resist Gaussian collapse.

### 5. Entropy and Dead Weight

| | Gemma 3 | G4 E4B | G4 26B MoE | G4 31B Dense |
|---|---|---|---|---|
| Shannon entropy (bits/byte) | 7.936 | 7.888 | 7.756 | 7.911 |
| Zero bytes | 0.71% | 0.39% | 0.68% | 0.51% |
| Near-zero Q4_K scales | 18.2% | 18.2% | **26.0%** | 19.9% |
| Low-entropy 4KB blocks | 3 | 21 | 21 | 0 |
| Zero regions (>256 bytes) | 0 | 2 | 0 | 0 |

The 26B has 26% near-zero scale blocks — the highest. This is intentional MoE sparsity: 120 of 128 experts are inactive per token, so their quantized blocks contribute almost nothing.

The 31B has zero low-entropy blocks and zero dead regions — the most uniformly utilized model. Every byte carries information.

The E4B's 21 low-entropy blocks (offsets 0x8f1e*) map to `v.position_embd.weight` — the vision position embeddings, which are inherently structured (nearby positions learn similar vectors).

### 6. Per-Layer Entropy Is U-Shaped (E4B)

Byte-level Shannon entropy of `ffn_gate.weight` across E4B blocks:

```
blk.0:  7.778  ████████████████████████████████████████
blk.10: 7.764  █████
blk.20: 7.763  ██                   ← tightest (most specialized)
blk.30: 7.776  ███████████████████████████████████
blk.41: 7.776  ███████████████████████████████████
```

Middle layers (blk.20) have the tightest weight distributions. Early layers handle diverse inputs, late layers handle diverse outputs, middle layers operate in the most compressed representational space. This matches the bottleneck gating architecture — the 2560→256→2560 compression is doing the most work in the middle.

### 7. Sliding Window Pattern: Every 6th Layer Is Global

Decoded from the E4B's `sliding_window_pattern` array (42 booleans):

```
SWA:    0,1,2,3,4,  6,7,8,9,10,  12,13,14,15,16, ...  (35 layers)
Global: 5,          11,           17,23,29,35,41        (7 layers)
```

Every 6th layer uses global attention (full 131K context, RoPE base 1M). The other 35 layers use sliding window attention (512 tokens, RoPE base 10K). This is a 5:1 local:global ratio — the model attends locally most of the time, reaching out to the full context only at regular intervals.

### 8. Vision Encoder: E4B Shrunk, Others Unchanged

| | Gemma 3 | G4 E4B | G4 26B | G4 31B |
|---|---|---|---|---|
| Vision blocks | 27 | **16** | 27 | 27 |
| Vision dim | 1152 | **768** | 1152 | 1152 |
| Vision FFN | 4304 | **3072** | 4304 | 4304 |
| Has bias | yes | **no** | yes | yes |
| QAT calibration | no | **yes** | no | no |
| FFN type | MLP (fc1/fc2) | **Gated (gate/up/down)** | MLP | MLP |

The E4B aggressively compressed the vision encoder (40% fewer blocks, 33% narrower) and compensated with quantization-aware training calibration tensors. The 26B and 31B kept the Gemma 3 vision architecture unchanged — Google considers this a solved design at the larger sizes.

Only the E4B adds an audio encoder (12 blocks, 1024 dim, conv kernel 5). The larger models are text+vision only.

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

The anomaly scan and frequency analysis are unique to `decompile.c`. No standard GGUF tool examines the actual weight data — they parse metadata and tensor manifests only.

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
