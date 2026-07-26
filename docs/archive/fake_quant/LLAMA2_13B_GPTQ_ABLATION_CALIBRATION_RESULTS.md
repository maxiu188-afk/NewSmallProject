# Llama-2-13B GPTQ component and calibration results

## Result

The completed CUDA study adds the missing naive GPTQ F3 control, a cumulative
QuaRot component ablation, and a 32/64/128-sequence calibration-size study.
All rows use the pinned `meta-llama/Llama-2-13b-hf` revision
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1`, pinned WikiText-2 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, W4A4, K/V at 16-bit, the same
331,614 test targets, symmetric group/block-128 GPTQ, 1% damping, activation
ordering, and seed 0. Every row records 40 decoder layers and 280 GPTQ linear
layers.

| Cumulative 128-sequence row | Mean NLL | PPL |
|---|---:|---:|
| Naive F3 (no rotation) | 9.062345 | 8624.350905 |
| Residual Hadamard | 2.875169 | 17.728424 |
| Residual + V/O | 3.426467 | 30.767735 |
| Residual + V/O + MLP | 1.765628 | 5.845242 |
| Complete F4 (+ post-RoPE Q/K) | 1.764316 | 5.837575 |

The same-run BF16 reference is PPL `5.008716` for every row. The naive F3
control is therefore catastrophic even with GPTQ, while full QuaRot F4 remains
within `0.828860` PPL of BF16.

| Calibration sequences | Naive F3 PPL | Complete F4 PPL |
|---:|---:|---:|
| 32 | 7427.883267 | 5.867919 |
| 64 | 8231.375585 | 5.847780 |
| 128 | 8624.350905 | 5.837575 |

## Interpretation and boundary

This is a cumulative, one-seed component sequence, not a full factorial
ablation. In that sequence, adding V/O after residual rotation worsens PPL by
`13.039311`, adding the MLP transform then reduces it by `24.922493`, and
post-RoPE Q/K improves the near-complete row by `0.007667` PPL. These are
path-dependent observations; they do not establish isolated causal effects or
rule out interactions between transformations.

Complete F4 is stable across the tested calibration budgets: relative to 128
sequences, 32 and 64 sequences add `0.030344` and `0.010205` PPL respectively.
The naive rows remain catastrophic and non-monotonic across calibration sizes,
so they do not support a claim that more calibration data improves this
portable naive GPTQ implementation.

All rows are floating-point fake quantization. They do not demonstrate packed
W4/KV storage, integer kernels, latency, throughput, or memory reduction.

## Runtime and artifact review

The study ran on an RTX 6000 Ada (48 GB), driver `550.127.05`, Python `3.12.3`,
PyTorch `2.6.0+cu124`, Transformers `5.14.1`, and Datasets `5.0.0`. Before the
13B matrix, a two-layer CUDA GPTQ gate processed all 14 linear layers with
finite post-quantization logits. The server wrote its preflight, smoke JSON,
study plan, nine result JSON files, and nine per-row logs under
`results/llama2-13b-wikitext2-gptq-study/`.

The raw JSON/log artifacts were copied from `/workspace` and SHA-256 checked:
all 18 corresponding files matched. Logs contain no OOM, CUDA, Hessian, or
traceback failure. Raw artifacts remain ignored by Git; see `results/README.md`.
