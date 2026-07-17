# Llama-2-13B matched RTN W4A4 results

## Result

The first formal low-bit text-evaluation pair completed on 2026-07-17. It
compares the portable pipeline's current RTN-style floating-point QDQ control
(F3) with its QuaRot reparameterized candidate (F4). This is not GPTQ,
packed-weight inference, low-bit KV-cache, a CUDA kernel, or a throughput
measurement.

| Row | Rotation | W/A/K/V bits | Mean NLL | PPL | Mean / max first-logit absolute error |
|---|---|---:|---:|---:|---:|
| F0 reference within these runs | none | 16 / 16 / 16 / 16 | 1.611151 | 5.008573 | 0 / 0 |
| F3 naive RTN W4A4 | none | 4 / 4 / 16 / 16 | 9.073338 | 8719.677539 | 2.290695 / 28.031250 |
| F4 QuaRot RTN W4A4 | residual, V/O, MLP, post-RoPE Q/K Hadamard | 4 / 4 / 16 / 16 | 2.522574 | 12.460633 | 1.188127 / 16.250000 |

QuaRot reduces PPL by `8707.216906` relative to the naive row and reduces the
mean/max first-logit error by about 48%/42%. F4 nevertheless remains about
2.49 times the matched BF16 reference PPL, which is the accuracy gap that the
later GPTQ stage must target.

## Matched protocol and provenance

| Item | Value |
|---|---|
| Model | `meta-llama/Llama-2-13b-hf` at `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` |
| Evaluation data | `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test` at `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Evaluation | 162 non-overlapping sequences of 2048 tokens; 331,614 next-token targets; seed 0 |
| Runtime | One RTX 6000 Ada (48 GB), driver `570.124.06`, compute capability `8.9` |
| Software | Python `3.11.15`, PyTorch `2.11.0+cu128`, Transformers `5.14.1` |
| Pipeline revision | `7d1ae46` |
| Artifacts | Ignored JSON files at `results/llama2-13b-wikitext2-rtn-w4a4/{naive-f3,quarot-f4,comparison}.json` on the network volume |

The comparison gate passed: it verified equal model revision, data revision,
split, sequence length, batch size, sample limit, produced batch count, seed,
and W4A4-with-16-bit-KV settings. The F4 result additionally records a
5120-wide residual transform (`40 x 128`) and a 13,824-wide online MLP
transform (`108 x 128`), both using the corresponding matrices from the
pinned upstream reference as portable data, not an upstream CUDA dependency.

## Next gate: GPTQ

Do not label these rows GPTQ. Before GPTQ, choose and pin a separate
calibration corpus, sample count, token count, group size, damping, act-order,
symmetry, seed, and bit widths. Then compare GPTQ F3/F4 against these RTN
rows and F0 under the same pinned WikiText-2 test protocol.
