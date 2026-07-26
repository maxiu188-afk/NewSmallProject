# Owned real-deployment plan

## Decision

The completed Llama-2-13B GPTQ study establishes an accuracy baseline only.
This plan starts a separate deployment track: real packed storage and low-bit
execution must be demonstrated before reporting memory, latency, or throughput.

`QuaRot/` remains a pinned source-level reference at
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`.  It is useful for recovering the
int4 layout, tensor boundaries, cache design, and test shapes.  Its Python,
build, CUDA, and benchmark code will not be imported, patched in place, or
treated as the implementation to deploy.

The first deployable target is the already pinned Llama-2-13B checkpoint on a
RunPod NVIDIA CUDA GPU.  The first execution path is **GPTQ W4A8**: packed
W4 with per-group scales, per-token symmetric A8, int32 accumulation, and
FP16/BF16 output.  It is deliberately separate from the more aggressive
W4A4KV4 fake-quant experiment.  KV4 comes only after W4A8 is numerically
verified end to end.

## What the upstream reference contributes, and what it does not

| Upstream code path | Reusable observation | Consequence for this project |
|---|---|---|
| `quarot/functional/quantization.py` | Signed two's-complement int4 packs two consecutive last-axis values into one `uint8`. | Define one owned, tested pack/unpack layout and persist its layout version in every checkpoint. |
| `quarot/nn/linear.py` and `kernels/gemm.cu` | The intended linear path is packed W4 x packed activation with int32 accumulation; the reference requires `K % 32 == 0`. | Start with Llama linear shapes satisfying the constraint, reject unsupported shapes explicitly, and compare the integer accumulator before dequantization. |
| `transformers/kv_cache.py` and `kernels/flashinfer.cu` | The cache is paged, stores K/V as packed `uint8`, and keeps FP16 scale/zero metadata. | Specify the cache layout independently and validate pack, append, and decode before connecting it to a model. |
| `e2e/quantized_llama/modeling_llama.py` | The deployment path replaces Llama linear modules and cache handling, rather than merely loading a low-bit checkpoint. | Build a maintained adapter against the selected current Transformers API; do not subclass the 2024 implementation. |
| `setup.py`, `requirements.txt`, and `e2e/benchmark.py` | It pins PyTorch 2.2.1/Transformers 4.38, targets only `sm_75/sm_80/sm_86`, invokes nested editable installs, and benchmarks with zero warm-up and one timed iteration. | Use a fresh, recorded modern CUDA environment; compile for the actual GPU capability; replace the build and benchmark harness entirely. |

The reference KV kernel currently enforces `head_dim == 128` and its cache
wrapper does not support batches with unequal page counts.  These are initial
supported-shape constraints, not silent assumptions.  Llama-2-13B satisfies
the head-dimension constraint; any generalisation must be implemented and
validated separately.

## Three execution phases

Each phase is a coherent work package, not a reason to repeatedly create and
tear down a server.  A phase may contain several local commits and checks.  On
one Pod, all server work uses one CUDA environment on the container disk;
`/workspace` retains only project code, model/Hugging Face caches, generated
artifacts, and the recorded environment specification.  A replacement Pod
creates a fresh container-disk environment from that saved specification; it
does not reconstruct an ad-hoc environment from memory or persist a virtual
environment on the shared volume.

| Phase | Owned work | Required evidence to complete | Claim allowed after completion |
|---|---|---|---|
| 1 — W4A8 execution foundation | Prepare the reproducible CUDA environment once, recording GPU, compute capability, driver, CUDA compiler, PyTorch CUDA build, Python, dependencies, and Git revision. Implement PyTorch CUDA-tensor int4 pack/unpack, groupwise W4 quantization, scale serialization, a dequantized matmul oracle, and the owned CUDA/CUTLASS-style W4A8 linear extension. Inputs are per-token symmetric A8; weights are packed W4; accumulation is int32; scales are applied after integer GEMM. | Saved preflight/build/import logs; bit-exact signed-int4 pack round trip; serialized weights reload identically; pre-dequant int32 accumulators exactly equal the independent oracle for all supported Llama shapes; scaled FP16/BF16 output meets a predeclared tolerance. | `int4_gemm` for the recorded linear shapes and GPU. |
| 2 — Llama deployment correctness | Build a maintained Llama adapter that loads a self-describing packed checkpoint and preserves the validated QuaRot/GPTQ rotation convention. Integrate the Phase-1 W4A8 linears first with BF16 K/V, then implement the owned paged KV4 packing, append, and decode-attention path. `head_dim=128`, page size, batch layout, and supported sequence lengths are explicit configuration fields. | Fixed-token layer logits/hidden states agree with the Phase-1 reference; generation smoke has finite outputs; KV pack/dequant, append, and page-boundary decode agree with an independent FP reference; manifest records actual precision at every linear and cache boundary. | End-to-end W4A8 execution; `int4_kv_cache` only for the tested cache layouts. |
| 3 — Deployment measurements | Benchmark only the Phase-2 paths that passed numerical gates, alongside a matched BF16 baseline. Measure linear, prefill, decode, and end-to-end generation separately. | Warm-up followed by repeated CUDA-synchronized samples; fixed workload grid and seeds; raw timings, mean/variation, tokens/s, peak allocated and reserved memory, configuration/implementation revisions, and linked correctness-run IDs. | GPU- and workload-specific performance and memory results. |

## Numerical contracts

Each gate writes a machine-readable result under `results/` and keeps raw logs
ignored by Git.  The result must include shapes, dtypes, quantization axes and
group size, packing order, scale dtype, GPU, compiler/runtime versions, random
seed, maximum absolute and relative errors, and an explicit pass/fail verdict.

The test hierarchy is intentionally strict:

1. Pack/unpack is bit-exact for all signed int4 values `[-8, 7]`.
2. Phase 1's pre-dequantization int32 accumulators are exactly equal to its
   integer oracle.
3. Phase-1 post-scale output is compared to the packed-W4 dequantization
   oracle at a declared
   tolerance chosen before the run and recorded per dtype.
4. Phase 2 compares fixed inputs against the immediately preceding verified
   reference; finite outputs alone never pass a correctness gate.

No timing command is permitted until the relevant Phase-1 or Phase-2 numerical result
exists and passes.  Fake-quant PPL, packed checkpoint size, and a successful
CUDA compilation are each insufficient on their own.

## Implementation boundaries

- New code lives in this repository (proposed `repro/deployment/`,
  `csrc/`, `scripts/`, `tests/`, and `configs/deployment/`); `QuaRot/` stays
  ignored and unchanged.
- The packed checkpoint format must have a JSON manifest containing model and
  source revisions, rotation/GPTQ parameters, group size, signed-int4 order,
  scale dtype/axis, tensor names/shapes, and SHA-256 checksums.
- Phase 1 is allowed to dequantize for an oracle, but that substep is labelled
  `packed_weight`; only its verified CUDA extension is `int4_gemm`.
- Phase 2 starts with W4A8 because it is a practical execution target.  W4A4 and
  KV4 are separate precision modes, configurations, numerical tests, and
  performance rows; success in one does not validate another.
- The current RTX 6000 Ada study environment is a suitable initial target
  only after Phase 1 records its actual compute capability and compatible toolchain.
  No local macOS result is deployment evidence.

## First implementation slice

Start Phase 1 locally, then use one scoped RunPod environment for the complete
Phase-1 CUDA portion:

1. Add the packed-W4 layout/specification and deterministic CPU/PyTorch tests.
2. Add a small export/import tool for one GPTQ-transformed linear tensor plus
   its manifest and checksums.
3. Prepare a versioned environment specification plus bootstrap script.  Build
   the virtual environment on container disk, and save the resolved package
   list and build output on the persistent volume so a replacement Pod can
   recreate the same environment.
4. In the same Phase-1 server session, run the CUDA-tensor packed-dequant
   numerical smoke and the W4A8 int4-GEMM accumulator/output checks for
   representative Llama-2-13B shapes (`5120 x 5120`, `13824 x 5120`, and
   `5120 x 13824`).

Phase 2 begins only after that single Phase-1 server package passes.  No model
PPL or performance collection starts before the relevant Phase-2 or Phase-3
entry condition is met.

This ordering keeps source-code archaeology, packed-checkpoint correctness,
native kernel correctness, and performance claims independently auditable.
