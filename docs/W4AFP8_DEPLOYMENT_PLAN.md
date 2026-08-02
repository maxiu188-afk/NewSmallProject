# QuaRot and SpinQuant W4AFP8 deployment plan

## Decision and sequence

This is a hardware-aligned extension for Isambard GH200, not a relabelling of
the papers' INT8-activation experiments. QuaRot-only validation job `5873544`
and its dependent SpinQuant-transfer job `5873545` have passed. Joint
four-model source-gate job `5874345` is queued to freeze the single provenance
record required by formal PPL and serving validation. The corrected SpinQuant
no-had fake-quant chain remains complete in its own checkout and artifact root.

The first SpinQuant no-had chain (`5850956` -> `5850958`) was accepted and is
retained as an old-QDQ baseline. QuaRot Gate 1 job `5854439` and SpinQuant
transfer Gate 1 job `5857916` were cancelled before allocation on 2026-08-01
after the GPTQ protocol was changed. Jobs `5859043`/`5859044` failed
same-revision calibration validation, and jobs `5873446`/`5873447` passed
calibration and backend capability checks but stopped before quantization on an
over-strict observer-alias assertion. None produced a deployment checkpoint or
result. The active order is now fixed:

1. retain accepted QuaRot W4AFP8 validation gate `5873544` from fix revision
   `9d691e9cf828`;
2. retain accepted isolated BF16/unrotated/SpinQuant-transfer W4AFP8 gate
   `5873545`, which ran through scheduler dependency `afterok:5873544`;
3. accept queued joint source-gate job `5874345`, then freeze its result and
   capability hashes into the formal PPL and serving configs;
4. evaluate deployed-checkpoint quality and serving performance only after the
   joint source gate has frozen checkpoint provenance;
5. treat the INT8-trained SpinQuant rotation as a transfer diagnostic and run
   FP8-targeted learning later if deployed quality is not competitive.

The W4AFP8 checkout and artifacts remain isolated from the SpinQuant fake-quant
redo. Within W4AFP8, the scheduler dependency gates allocation only: the
QuaRot-only mode does not read the learned rotation, and the SpinQuant mode
must pass its own packed-checkpoint and inference checks. The existing W4A8
fake-quant result answers the paper-aligned INT8 quality question only. It
cannot be reused, converted, or extrapolated into a W4AFP8 accuracy result
because FP8 changes the activation quantizer, error distribution,
calibration/learning objective, and executed kernel.

The online-Hadamard `SpinQuant_had` path remains a separate paper extension. It
is not a dependency for the offline-only W4AFP8 deployment defined here.

## Frozen runtime target

The first formal target is Llama-2-13B on one Isambard GH200, using the already
accepted vLLM `0.25.1+cu129` serving environment and a compatible LLM
Compressor export environment. The intended common runtime form is:

- symmetric group-128 INT4 decoder weights;
- GPTQ without activation ordering and with ordinary min/max weight ranges; no
  additional MSE weight clipping;
- dynamic per-token FP8 activations, with the exact FP8 dtype and scale contract
  recorded from the selected vLLM kernel;
- BF16 outputs, normalization, residual arithmetic, embeddings, and `lm_head`;
- 16-bit KV cache for the first study;
- all rotations fused into standard Llama parameters offline;
- no online R3/R4 Hadamard transforms and no custom attention or KV-cache code;
- no runtime activation-order `g_idx` if the selected Hopper W4AFP8 kernel does
  not support it.

Group size 128 is a backend constraint, not an accuracy tuning choice: the
pinned `CutlassW4A8LinearKernel` rejects any other group size. Consequently
this real-deployment GPTQ cannot be made identical to the paper-oriented
`group_size=-1` protocol. It is instead the weaker option still executable by
the selected accelerated backend: group-128 with activation ordering disabled.
Its PPL must be compared only across the matched W4AFP8 variants, not directly
against fake-quant rows produced by a different GPTQ protocol.

Every decoder projection must remain covered after packing. The exporter must
record the 280 expected packed linears, validate all Llama-2-13B matrix shapes
against the kernel's alignment constraints, and reject a silent fallback to
BF16, W4A16, or another mixed-precision kernel.

## Compared variants

| Variant | Offline transform | Accepted evidence required |
|---|---|---|
| BF16 reference | None | Same model revision, tokens, requests, GPU, and serving flags |
| Unrotated W4AFP8 control | None | Deployed checkpoint quality and real W4AFP8 serving |
| QuaRot-style W4AFP8 | Offline-fusible residual and V/O rotations used by the completed W4A16 route | Deployed checkpoint quality and real W4AFP8 serving |
| SpinQuant W4AFP8 | Learned offline R1/R2 | Deployed checkpoint quality and real W4AFP8 serving |

The accepted SpinQuant endpoint must optimize or validate rotations against an
FP8-activation objective. Existing W16A8 rotations may be tested as a clearly
labelled transfer diagnostic, but that diagnostic is not automatically the
final SpinQuant W4AFP8 result. If transfer quality is not competitive with the
unrotated control, an FP8-targeted rotation-learning run is required before the
formal comparison.

## Execution gates

### Gate 0: dependency and provenance

- The corrected fake-quant jobs are terminal and accepted.
- Verify same-revision execution, exit codes, stage markers, result hashes, and
  the full 162 x 2048 held-out-token result before syncing or executing W4AFP8.
- Keep the accepted INT8 W4A8 artifacts immutable and separately named.

### Gate 1: W4AFP8 numerical and export smoke

- Add an FP8 activation simulation only as an implementation diagnostic; do not
  publish its PPL as deployed W4AFP8 accuracy.
- Export tiny and then full Llama checkpoints for unrotated, QuaRot-style, and
  SpinQuant variants with identical W4AFP8 metadata.
- Validate save/reload behavior, quantization metadata, packed-linear coverage,
  scale finiteness, excluded-module precision, and absence of unsupported
  runtime `g_idx` state.
- Load each checkpoint in a fresh process and prove that vLLM selects the
  intended Hopper W4AFP8 kernel rather than a fallback path.

### Gate 2: deployed-checkpoint accuracy

Accuracy is measured from the exported checkpoints through the deployment
stack, not from the existing QDQ evaluator:

- use the same retained 162 x 2048 WikiText-2 test-token artifact, with 331,614
  scored next-token targets;
- run BF16, unrotated W4AFP8, QuaRot-style W4AFP8, and SpinQuant W4AFP8 in fresh
  child processes under one pinned evaluation protocol;
- report total NLL and PPL, plus fixed-token logits/greedy-token checks before
  the full pass;
- record checkpoint hashes, runtime versions, selected kernel, request failures,
  and token counts;
- compare every rotated result with both BF16 and the unrotated W4AFP8 control.

No INT8 W4A8 fake-quant PPL, W4A16 PPL, fixed-token smoke, or layer-only result
may substitute for this gate.

### Gate 3: service correctness and acceleration

Use the completed W4A16 serving protocol so that the new result is comparable:

- one GH200; identical server flags and fixed 8 GiB KV-cache allocation;
- 256 input tokens and 64 output tokens;
- concurrency 1 and 8;
- 4 warm-up requests and 64 measured requests per case;
- `/health`, `/v1/models`, and deterministic completion checks before timing;
- 64/64 completed requests, zero failures, and post-shutdown GPU-memory recovery.

For BF16 and all three W4AFP8 variants, record startup time, ready/peak GPU
memory, TTFT, TPOT, end-to-end latency p50/p95, request throughput, and token
throughput. Report speedup only when the logs prove that the real W4AFP8 kernel
executed. Fake-quant wall time, checkpoint size, and a single-layer benchmark
are not full-model acceleration evidence; a layer diagnostic may be retained
only as secondary explanation.

### Gate 4: acceptance and reporting

A variant is deployment-complete only after it passes both Gate 2 and Gate 3.
The formal deployment experiment therefore consists of two required formal
jobs after their respective smokes:

1. a deployed-checkpoint accuracy job that records total NLL, PPL, scored-token
   count, and PPL deltas against both BF16 and unrotated W4AFP8;
2. a matched full-model serving job that records throughput, TTFT, TPOT,
   end-to-end latency, GPU memory, request success/failure counts, and the
   selected W4AFP8 kernel.

Neither formal job can substitute for the other. A successful serving speedup
without deployed-checkpoint PPL is an incomplete performance result, while PPL
without the serving benchmark is an incomplete quality result.
The final comparison must present the quality/performance trade-off for BF16,
unrotated W4AFP8, QuaRot-style W4AFP8, and SpinQuant W4AFP8, while keeping the
completed QuaRot-style W4A16 result as a separately named reference point.

The final claims are bounded as follows:

- `no_had W4A8 fake quant`: paper-aligned floating-QDQ quality evidence;
- `QuaRot-style W4AFP8`: offline-rotation, hardware-aligned deployment extension;
- `SpinQuant W4AFP8`: learned offline-rotation, hardware-aligned deployment
  extension;
- `acceleration`: only matched full-model serving evidence from the selected
  W4AFP8 backend.

## Current execution boundary

The prior QuaRot gate `5854439` and SpinQuant-transfer gate `5857916` are
cancelled with zero runtime because they carried the superseded static
activation-order protocol. The two subsequent failed pairs are diagnostic only:
the first exposed stale calibration provenance, while the second proved the
calibration and backend audit before exposing the observer-alias check. The
corrected SpinQuant rotation provenance remains frozen. Validation job
`5873544` and dependent main job `5873545` completed from revision
`9d691e9cf828`; both use the same group-128/no-actorder configuration. They
produced the three expected compressed checkpoints and passed isolated vLLM
load/inference gates. Joint four-model source-gate job `5874345` is queued from
the same revision because later validators require one complete result and
capability hash. These Gate 1 jobs record neither the formal 331,614-token PPL
result nor the formal serving speedup. No deployment evidence is accepted
until the joint gate artifacts pass provenance checks and the later accuracy
and serving formal jobs both pass. The first SpinQuant checkpoint remains an
INT8-trained rotation transfer diagnostic, not an FP8-optimized endpoint.

The QuaRot-only Gate 1 path remains isolated and does not read the SpinQuant
rotation. Deployed PPL and serving still require accepted gate results and
frozen hashes before submission.
