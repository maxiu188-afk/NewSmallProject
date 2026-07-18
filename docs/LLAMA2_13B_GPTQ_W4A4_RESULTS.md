# Llama-2-13B QuaRot GPTQ W4A4 result

## Result

The first formal QuaRot + GPTQ F4 W4A4 evaluation completed successfully on
2026-07-17. It uses the pinned `meta-llama/Llama-2-13b-hf` revision
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` and `Salesforce/wikitext`,
`wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

| Row | Weight fitting | Residual rotation | W/A/K/V | Mean NLL | PPL | Targets |
|---|---|---|---|---:|---:|---:|
| F0 reference in this run | none | none | 16/16/16/16 | 1.611180 | 5.008716 | 331,614 |
| F4 QuaRot RTN | RTN QDQ | Hadamard | 4/4/16/16 | 2.522574 | 12.460633 | 331,614 |
| F4 QuaRot GPTQ | GPTQ | Hadamard | 4/4/16/16 | 1.764316 | 5.837575 | 331,614 |

Against the matched QuaRot RTN result, GPTQ reduced PPL by `6.623058`
(`53.15%`). The GPTQ result is `0.828860` PPL above its same-run BF16
reference. This is a substantial accuracy improvement, not a deployment or
throughput result.

## Protocol and runtime

- Runtime: RTX 6000 Ada (48 GB), driver `550.127.05`, Python 3.12, PyTorch
  `2.6.0+cu124`, Transformers `5.14.1`.
- Evaluation: 162 sequential, non-overlapping `test` sequences of length 2048
  (331,614 next-token targets).
- Calibration: the pinned `train` split, 128 complete sequences of length 2048
  (262,144 input tokens), read deterministically in source order.
- GPTQ: symmetric W4, group size 128, block size 128, 1% Hessian damping, and
  activation ordering enabled. It processed all 40 decoder layers and 280
  linear layers.
- QuaRot: residual Hadamard, V/O and online structured MLP Hadamards,
  post-RoPE Q/K rotation; activations are A4 and K/V remain at 16-bit.

The saved JSON contains finite metrics and a calibration record for each
linear layer. Its first-logit mean/max absolute error against the same-run
reference is `0.637860` / `9.703125`.

## Boundary and next comparison

This is a floating-point fake-quantization experiment. It does not pack W4
weights, use integer kernels, quantify the KV cache, or measure latency,
throughput, or memory reduction.

The matched naive GPTQ F3 control has not yet been run. Therefore this record
establishes that GPTQ materially improves the QuaRot F4 path over RTN, but it
does not yet isolate the incremental contribution of QuaRot under GPTQ. The
next full-GPU comparison, when resources are available, is
`llama2_13b_wikitext2_naive_w4a4_gptq.json` with the identical calibration and
test protocol.

## Artifact provenance

The downloaded raw artifacts are intentionally ignored by Git and stored
locally under `results/llama2-13b-wikitext2-gptq-w4a4/`:

- `quarot-f4-gptq.json`
- `quarot-f4-gptq.log`

They were copied from the RunPod persistent volume and SHA-256 checked against
the server before shutdown. See `results/README.md` for the generated-artifact
policy.
