# QuaRot Reproduction Plan

## 1. Scope and principles

This project reproduces **QuaRot only**. `Spin.pdf` and related work are out of
scope unless a clearly documented, method-independent reference is needed.

`QuaRot/` is the upstream reference implementation, pinned locally at commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`. It is used to understand intended
behaviour, tensor placement, checkpoint format, and CUDA execution paths. It
is not treated as code to copy wholesale: its Python, Transformers, CUDA, and
benchmark assumptions are from 2024 and must be validated before reuse.

The reproduction has two strictly separated tracks:

| Track | Question answered | Allowed claim |
|---|---|---|
| Algorithm / fake quant | Does rotation preserve the model and improve low-bit QDQ accuracy? | Algorithmic simulation result |
| Deployment / kernels | Does a real low-bit kernel reduce storage or improve measured serving cost? | CUDA deployment result for the exact tested configuration |

No result may be labelled “A4W4KV4 deployment” merely because a model loads or
generates text. The report must state whether weights, activations, KV cache,
and matrix multiplications are genuinely low-bit or only simulated.

## 2. Execution environments

### Current scope decision (2026-07-15)

The active reproduction first completes algorithm implementation and local
Windows GPU smoke tests. Only after those gates pass may a server be used for
pretrained-model PPL, GPTQ calibration, or large controlled matrices. CUDA
kernel builds and performance claims remain separately gated deployment work;
they must not be inferred from local fake-quant results.

If the server scope is reopened, `docs/CUDA_PYTORCH_COMPATIBILITY_POLICY.md` is
mandatory before any dependency or build script is written.

### Local Windows GPU: correctness and smoke tests only

- Platform: Windows with an NVIDIA GPU; the currently observed device is an
  RTX 3070 Ti Laptop GPU with 8 GB VRAM. Exact driver, PyTorch CUDA build, and
  device information must be recorded by each smoke result.
- Use: deterministic mathematical tests, tiny/random model tests, static source
  audit, config/schema validation, and small CUDA correctness smoke tests.
- Common development packages may be installed locally when needed. Defer
  Datasets, Accelerate, LM-Eval, CUDA extensions, and model-weight downloads
  until there is a concrete need and the dependency scope has been reviewed.
  The approved exception is the isolated local smoke environment defined in
  `requirements-local-smoke.txt`: PyTorch and Transformers only, used solely
  for randomly initialized tiny-model equivalence tests.
- Do not use it for 7B-scale PPL, GPTQ calibration, CUDA extension compilation,
  or performance conclusions.

### RunPod GPU: reproducibility and performance experiments

**Runs only after the local GPU smoke gate passes.**

- Use: model loading, calibration, PPL and zero-shot evaluation, CUDA extension
  builds, kernel correctness, memory, prefill, decode, throughput, and latency.
- Each run records GPU model/count, driver, CUDA toolkit, PyTorch, Transformers,
  Python, Git revision, command, seed, model revision, and dataset revision.
- Start with one supported LLaMA-2 configuration and a small smoke workload;
  scale only after the preceding gate passes.

## 3. Planned project layout

New reproduction code will live beside, not inside, `QuaRot/`:

```text
configs/                 versioned experiment definitions
repro/                   small, documented implementation and adapters
scripts/                 common local/RunPod entry points
tests/                   deterministic correctness and smoke tests
docs/                    method notes, compatibility notes, result summaries
results/                 ignored generated logs, metrics, tables, checkpoints
patches/                 minimal, reviewable patches against upstream if needed
QUAROT_REPRODUCTION_PLAN.md
QuaRot/                  upstream reference; avoid unreviewed modification
```

Configuration is the source of truth for an experiment. Every result directory
must contain a resolved config, environment snapshot, raw stdout/stderr, and
machine-readable metrics. Generated data, checkpoints, model weights, and
large logs are not committed.

## 4. Phases and acceptance gates

### Phase 0 — Reproduction contract and environment audit

**Goal:** make every later result attributable and comparable.

**Work**

- Record the upstream commit/submodule revisions and the observed code paths.
- Define a common experiment schema: model, precision, rotation settings,
  quantizer settings, calibration data, evaluation data, seed, and runtime.
- Add result labels that distinguish `bf16`, `fake_quant`, `packed_weight`,
  `int4_gemm`, and `int4_kv_cache`.
- Create separate local and RunPod environment manifests; do not install the
  upstream CUDA extension on the Mac.

**Checkable exit criteria**

- A configuration can be validated without downloading a model.
- A generated run manifest contains all required provenance fields.
- The official source remains unchanged; any future compatibility change first
  appears as a focused patch under `patches/` with a reason and verification.

### Phase 1 — Quantization and rotation primitives (local)

**Goal:** verify the mathematics independently of old framework code.

**Work**

- Implement small, device-agnostic reference functions for symmetric and
  asymmetric quantize/dequantize, signed int4 pack/unpack, and normalized
  Hadamard transforms.
- Check orthogonality, inverse recovery, quantizer range/scale behaviour, and
  packed representation round trips.
- Compare valid cases against the corresponding upstream functions, without
  importing or copying their CUDA-only execution path.

**Checkable exit criteria**

- Deterministic tests pass on CPU; optional MPS results are reported separately.
- Hadamard round-trip and int4 pack/unpack are exact where mathematically
  expected; quantization error bounds are asserted.
- Tests cover dimensions relevant to LLaMA hidden, MLP, and head dimensions.

### Phase 2 — Model transformation equivalence (local first, RunPod confirmation)

**Goal:** establish that QuaRot itself does not change the full-precision model.

**Work**

- Test residual-stream rotation, layer-norm fusion, MLP down-projection online
  Hadamard rotation, V/O rotation, and Q/K rotation separately on toy modules.
- Run a tiny randomly initialized LLaMA-shaped model locally.
- On RunPod, repeat on the selected pretrained LLaMA-2 model with quantization
  disabled.

**Checkable exit criteria**

- Fixed-input logits and selected hidden states match an unrotated BF16/FP16
  reference within documented dtype-specific tolerances.
- The no-quantization PPL delta is negligible and reported before any W4 result.
- A failed equality check blocks subsequent quantization experiments.

### Phase 3 — Fake-quant accuracy experiments (RunPod)

**Goal:** reproduce the algorithmic benefit of rotation under controlled QDQ.

**Work**

- Use the same base model, evaluation corpus, calibration corpus, calibration
  sample count, sequence length, seed, and GPTQ/RTN settings in each comparison.
- Begin with a small smoke configuration, then run the following controlled
  matrix on LLaMA-2:

| ID | Rotation | Quantization scope | Role |
|---|---:|---|---|
| F0 | no | BF16/FP16 | accuracy baseline |
| F1 | no | W4 | naive weight-quantization control |
| F2 | yes | W4 | rotation effect on weights |
| F3 | no | W4A4 | naive end-to-end QDQ control |
| F4 | yes | W4A4 | core QuaRot fake-quant result |
| F5 | yes | W4A4KV4 | full algorithmic simulation |
| F6+ | partial | selected rotation disabled | ablation |

- Keep calibration and final PPL evaluation data separate when possible; record
  any deliberate use of the upstream default instead of concealing it.

**Checkable exit criteria**

- Every row has a complete manifest and raw PPL log.
- F0/F1/F2 and F0/F3/F4 are directly comparable pairs.
- The summary explicitly says `fake quantization`; no latency or throughput is
  inferred from these runs.

### Phase 4 — CUDA compatibility and kernel correctness (RunPod)

**Goal:** determine what portion of the upstream real-int4 implementation runs
correctly on the rented GPU.

**Work**

- Initialize upstream submodules only on RunPod and attempt a pinned build.
- Record the original build result before changing anything. The upstream setup
  currently targets `sm_75`, `sm_80`, and `sm_86`; GPU architecture mismatch is
  a compatibility finding, not a reason to silently replace code.
- If needed, make only a minimal, isolated architecture/build patch and compare
  results with the original intended path.
- Validate packed W4 linear output against a dequantized/reference calculation,
  then validate int4 KV cache attention against FP16 cache output.

**Checkable exit criteria**

- The build log and exact CUDA stack are saved.
- Kernel outputs pass documented numerical tolerance tests for every supported
  shape; unsupported shapes are listed explicitly.
- The report states kernel constraints such as head dimension, batch/page layout,
  and attention implementation.

### Phase 5 — Real deployment and performance measurements (RunPod)

**Goal:** measure only verified low-bit execution paths.

**Work**

- Benchmark FP16/BF16 and the verified QuaRot path separately for linear,
  KV-attention, layer-level, and end-to-end generation workloads.
- Measure prefill and decode separately across fixed batch size and sequence
  length grids. Include warm-up, CUDA synchronization, repeated samples, mean,
  variability, tokens/s, and peak allocated memory.
- Verify the checkpoint conversion and model execution path used in every
  benchmark before collecting timings.

**Checkable exit criteria**

- Each benchmark identifies the actual precision of weights, activations, GEMM,
  KV cache, and accumulation/dequantization.
- Results contain raw repeated measurements, not just one timing.
- Memory, latency, and throughput claims apply only to the recorded GPU and
  workload; fake-quant results remain in a separate table.

### Phase 6 — Research package and presentation material

**Goal:** prepare an auditable result for Professor Hu.

**Work**

- Produce concise method notes with transformation diagrams and quantization
  locations.
- Produce separate tables for equivalence, fake-quant accuracy, kernel
  correctness, and real deployment performance.
- Write a limitations section covering unavailable hardware, unsupported paths,
  version/architecture patches, failed experiments, and unverified paper claims.

**Checkable exit criteria**

- A new user can reproduce each completed result from a config and command.
- Claims in the summary link to saved raw artifacts.
- Completed, partial, and unattempted items are visibly separated.

## 5. Immediate next milestone

The local deterministic test suite and experiment-manifest validator are the
current final deliverables. No large-model dependency or RunPod environment is
needed for the agreed scope.

The next local-only extension is the portable LLaMA pipeline documented in
`docs/PORTABLE_PIPELINE.md`. It must remain model/data configuration driven and
run its first checks on random/synthetic inputs before any selected pretrained
model is downloaded.
