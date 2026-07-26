# Consolidated official QuaRot backend results

## Scope

This document combines the two accepted results from the pinned upstream
QuaRot backend:

1. a complete Llama-2-13B W4A4KV4 engineering extension; and
2. the paper-protocol-aligned Llama-2-7B single-decoder-block matrix.

These are real packed-weight, integer-accumulator, and KV4 execution results,
not the repository's earlier floating-point QDQ simulation. The full-model
extension and the single-block paper scope answer different questions and must
not be reported as one performance benchmark.

## Frozen environment

| Item | Recorded value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada, compute capability 8.9, 48 GB |
| Upstream QuaRot | `5008669b08c1f11f9b64d52d16fddd47ca754c5a` |
| Full model | `meta-llama/Llama-2-13b-hf` |
| Single block | `meta-llama/Llama-2-7b-hf` |
| Official runtime | PyTorch 2.2.1+cu121; Transformers 4.38.0; FlashAttention 2.5.6 for the block matrix |

Compatibility patches adapt SM89 build flags and framework interfaces. They do
not replace the upstream quantization, GEMM, or KV-cache algorithms.

## Complete Llama-2-13B result

The exported checkpoint contains all 280 decoder projections as upstream
`Linear4bit` modules, uses the upstream W4 activation path, and stores paged
K/V at 4 bits. The two-shard checkpoint is 6.6 GB and contains 280 `uint8`
packed tensors. The fixed-token smoke passed module-count, packed-weight,
finite-scale, finite-logit, cache-length, and exact repeated-generation gates.

### Matched batch-one performance

| Workload | Metric | FP16 | W4A4KV4 | FP16/W4 | Interpretation |
|---|---:|---:|---:|---:|---|
| 128 prefill + 8 decode | Prefill | 53.38 ms | 71.14 ms | 0.750x | W4 1.33x slower |
| same | Decode | 53.67 ms/token | 77.73 ms/token | 0.690x | W4 1.45x slower |
| same | End to end | 485.78 ms | 698.53 ms | 0.695x | W4 1.44x slower |
| 2048 prefill + 32 decode | Prefill | 392.51 ms | 441.70 ms | 0.889x | W4 1.13x slower |
| same | Decode | 52.28 ms/token | 79.51 ms/token | 0.658x | W4 1.52x slower |
| same | End to end | 2076.90 ms | 2945.97 ms | 0.705x | W4 1.42x slower |

### Memory at 2048 prefill + 32 decode

| Measurement | FP16 | W4A4KV4 | W4/FP16 |
|---|---:|---:|---:|
| Model resident | 26.292 GB | 7.179 GB | 27.30% |
| Prefill peak | 30.221 GB | 8.635 GB | 28.57% |
| Decode peak | 28.490 GB | 8.183 GB | 28.72% |
| End-to-end peak | 28.796 GB | 8.395 GB | 29.15% |

The supported full-model conclusion is a large capacity improvement, not a
latency speedup on this GPU at batch one. The old backend launches separate
activation-reduction, packing, GEMM, dequantization, Hadamard, and KV-cache
operations; those costs dominate the small decode matrices. No official packed
checkpoint PPL has been measured, so the fake-quant PPL must not be reused as a
deployment-quality result.

## Paper-aligned Llama-2-7B single-block result

The block runner retains one upstream decoder layer, seven W4A4 projections,
the KV4 cache, and the official 3-warm-up / 10-timed-call / 10-repeat structure.
W4A4KV4 completed all 14 formal cases. FP16 completed 13 and retained a
capacity OOM at batch 64, sequence length 2048. This is protocol-aligned rather
than a numerical table reproduction because the paper used an RTX 3090.

### Prefill at sequence length 2048

| Batch | FP16 mean | W4 mean | FP16/W4 speed | FP16 peak | W4 peak |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.46 ms | 3.88 ms | 1.667x | 1.239 GB | 0.373 GB |
| 4 | 29.18 ms | 17.34 ms | 1.683x | 3.681 GB | 1.158 GB |
| 16 | 117.61 ms | 76.98 ms | 1.528x | 13.483 GB | 4.297 GB |
| 64 | OOM | 320.91 ms | n/a | OOM after 44.39 GiB | 16.855 GB |

### Decode and layer end to end

At batch one, W4 decode remains slower across contexts 256--4096, with
FP16/W4 speed ratios of 0.58--0.66x. At batch 16 the gap narrows as context
grows: at context 2048, layer end to end reaches 1.024x; at context 4096,
decode-50 reaches 1.028x and layer end to end reaches 1.282x. Every paired
point uses less peak allocated memory with W4A4KV4; the E2E memory advantage
ranges from 1.87x to 3.28x.

The single-block result explains why the full model can save memory without
accelerating batch-one serving: the legacy kernel becomes useful for larger
prefill matrices and high-batch, long-context work, but small autoregressive
decode remains launch- and occupancy-limited. The crossover must not be
extrapolated to complete-model serving without measurement.

## Detailed archived records

- [`OFFICIAL_QUAROT_W4A4_RESULTS.md`](archive/results/OFFICIAL_QUAROT_W4A4_RESULTS.md)
- [`OFFICIAL_QUAROT_SINGLE_BLOCK_RESULTS.md`](archive/results/OFFICIAL_QUAROT_SINGLE_BLOCK_RESULTS.md)

The archived files retain exact artifact hashes, full provenance, raw-sample
summaries, and the detailed explanation of the negative batch-one result.
