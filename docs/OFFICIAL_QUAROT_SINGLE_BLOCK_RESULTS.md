# Official QuaRot single-block result

## Outcome

The paper-protocol-aligned Llama-2-7B decoder-block benchmark completed on one
NVIDIA RTX 6000 Ada. The pinned upstream W4A4KV4 path completed all 14 formal
cases. FP16 completed 13 cases and exceeded the 48 GB device capacity only at
the paper's largest prefill point, batch 64 at sequence length 2048. That OOM
is retained as a capacity result; the workload was not reduced.

The latency result depends strongly on shape. W4A4KV4 was 1.53--1.68x faster
for the three paired 2048-token prefill batches. It remained slower for
batch-one decode. At batch 16 and long context, the gap reversed: at context
4096, W4A4KV4 was 1.03x faster for decoding 50 tokens and 1.28x faster for the
layer-e2e prefill-plus-50-token workload. Every paired point used less peak
allocated memory with W4A4KV4.

This is the single-transformer-block performance scope used by the QuaRot
paper, not complete-model latency or quality. The paper used RTX 3090, so the
Ada result is protocol-aligned rather than a numerical reproduction of the
paper's tables.

## Frozen provenance

| Item | Recorded value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, compute capability 8.9, 49,140 MiB |
| Driver / host toolkit | 570.195.03 / CUDA 12.8.93 |
| Python / PyTorch runtime | 3.10.18 / 2.2.1+cu121 |
| Transformers / FlashAttention | 4.38.0 / 2.5.6 |
| Upstream QuaRot | `5008669b08c1f11f9b64d52d16fddd47ca754c5a` |
| Project base revision | `a16c3d66f943b7ecd97487310dd6cc2833087f0b` plus the benchmark working tree published with this result |
| Model | `meta-llama/Llama-2-7b-hf` |
| Model revision | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| Source adapters | SM89 build, Transformers-4.38 cache, Transformers-4.38 RoPE |
| Formal protocol | 3 warm-ups, 10 timed calls per repetition, 10 repetitions |
| Formal result status | `completed_with_oom` |

The two required safetensors blobs were verified against their SHA-256 object
names before loading. The source gate also verified the three patched-file
hashes and all three upstream submodule revisions. Each W4 layer contained
exactly seven upstream `Linear4bit` projections; the FP16 layer contained zero.
The formal JSON records the pre-commit project base plus the exact dirty-path
manifest because the benchmark package was synchronized to RunPod before this
publication commit.

As in upstream `e2e/benchmark_layer.py`, the W4 timing path constructs a
shape-correct packed decoder layer rather than loading a quantized quality
checkpoint; the pinned model revision supplies the architecture and the FP16
comparator weights. Weight values do not change the measured GEMM shapes, but
these timings must not be presented as a quality or perplexity result.

Before timing, the official primitive, KV4, and tiny random one-layer paths all
passed. The primitive int32 accumulator and scaled output were exact. KV4
prefill/append storage was exact and the decode output matched its dequantized
PyTorch reference within the frozen tolerance. Tiny W4A4KV4 prefill and decode
were finite and exactly repeatable.

## Prefill at sequence length 2048

Times are arithmetic means over the ten retained repetition samples. Peak
memory is CUDA peak allocated memory for the measured single-layer workload.
`FP16/W4` greater than one means W4A4KV4 is faster or uses less memory.

| Batch | FP16 mean ms | W4 mean ms | Speed FP16/W4 | FP16 peak GB | W4 peak GB | Memory FP16/W4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.46 | 3.88 | 1.667x | 1.239 | 0.373 | 3.32x |
| 4 | 29.18 | 17.34 | 1.683x | 3.681 | 1.158 | 3.18x |
| 16 | 117.61 | 76.98 | 1.528x | 13.483 | 4.297 | 3.14x |
| 64 | OOM | 320.91 | n/a | OOM after 44.39 GiB allocated | 16.855 | n/a |

The batch-64 FP16 call failed while requesting another 2.69 GiB with 44.39 GiB
already allocated by PyTorch. The W4A4KV4 point completed with a 16.86 GB peak.

## Decode and layer-e2e

`decode-50` times the 50 autoregressive decoder-block calls after an untimed
prefill. `E2E` times one decoder-block prefill followed by those 50 calls. The
last column compares peak allocated memory for E2E.

| Batch | Context | FP16 decode-50 ms | W4 decode-50 ms | Decode FP16/W4 | FP16 E2E ms | W4 E2E ms | E2E FP16/W4 | E2E memory FP16/W4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 64.50 | 109.30 | 0.590x | 66.66 | 108.72 | 0.613x | 3.28x |
| 1 | 512 | 65.95 | 113.37 | 0.582x | 66.85 | 114.75 | 0.583x | 3.04x |
| 1 | 1024 | 68.67 | 104.80 | 0.655x | 68.94 | 111.45 | 0.619x | 2.72x |
| 1 | 2048 | 68.48 | 109.64 | 0.625x | 71.11 | 108.92 | 0.653x | 2.44x |
| 1 | 4096 | 68.50 | 106.77 | 0.642x | 77.85 | 113.35 | 0.687x | 2.18x |
| 16 | 256 | 69.23 | 110.60 | 0.626x | 76.02 | 112.56 | 0.675x | 2.20x |
| 16 | 512 | 65.71 | 105.86 | 0.621x | 89.40 | 121.76 | 0.734x | 2.04x |
| 16 | 1024 | 71.79 | 109.94 | 0.653x | 115.23 | 140.49 | 0.820x | 1.94x |
| 16 | 2048 | 85.46 | 106.27 | 0.804x | 186.55 | 182.26 | 1.024x | 1.89x |
| 16 | 4096 | 116.38 | 113.19 | 1.028x | 350.47 | 273.34 | 1.282x | 1.87x |

The result reconciles the earlier full-model observation with the paper's
single-block scope. This legacy upstream backend does not accelerate small
batch-one decode on Ada, but its lower memory traffic becomes useful for
prefill and for high-batch, long-context layer-e2e work. The result does not
imply that full-model serving will achieve the same crossover.

## Completeness and artifacts

- W4A4KV4: 14/14 cases completed;
- FP16: 13/14 cases completed, with one retained capacity OOM;
- paired comparisons: 13 cases and 23 metrics;
- successful mode-metrics: 47, each with ten positive finite timing samples
  and ten positive peak-memory samples;
- smoke and formal outputs were copied back under the ignored local directory
  `results/quarot-official-single-block-rtx6000ada/`.

| Artifact | SHA-256 |
|---|---|
| `formal.json` | `ad4f1b42f9db9db081e26e66507349c51270898227ad121455f29435d0999402` |
| `formal.log` | `345f3ba0c2da173b82b8c4209c657bd49b4a6bb3cf0fcf0e8b58b257065e5485` |
| `smoke.json` | `d6fdd96c29e294538e36f915987019b63b4d53422536925a9bd7f9e3fa1417ba` |
| `primitive.json` | `b7da33d1aeef7c50db6da69a39d4f9e3f0a38e449d2eb8985a1b606552e16c84` |
| `kv.json` | `dca7bcd198b89c5be2e22d1e2128c799bdc45ea6d139b02585b40e4966c13a48` |
| `tiny-e2e.json` | `83f0b0db724e3883e19ec33749c9e37939e9a717a1163b27306d88ac43e932e2` |
| `preflight.json` | `36fe24646fc060079cd75449df2c36cbf000486f9c4b2ab8f21e6e365c35edb1` |
| `build.log` | `b86daff651bf85b9e9262f15f30219d8333d10413d6e4d4e2364f006f59a9583` |

The first formal attempt, which stopped at the batch-64 FP16 OOM before the
runner learned to continue, is retained separately as `formal-attempt1.*`.
