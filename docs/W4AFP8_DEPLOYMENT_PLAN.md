# QuaRot and SpinQuant W4AFP8 deployment plan

## Decision and sequence

This is a hardware-aligned extension for Isambard GH200, not a relabelling of
the papers' INT8-activation experiments. QuaRot-only validation job `5873544`,
its dependent SpinQuant-transfer job `5873545`, and joint four-model source gate
`5874345` have passed. The joint provenance hashes are frozen in revision
`1f3e4cb`; PPL smoke `5874806` and formal job `5874807` completed `0:0` through
`afterok:5874806`. The corrected SpinQuant no-had fake-quant chain remains
complete in its own checkout and artifact root.

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
3. retain accepted joint source-gate job `5874345` and its result/capability
   hashes;
4. retain accepted PPL smoke `5874806` and dependent formal job `5874807` over
   all 331,614 scored targets;
5. retain serving revision `ad971f9`, accepted result-gated service smoke
   `5882787`, accepted benchmark formal `5883004`, and their reviewed JSON/log
   hashes;
6. retain accepted BoolQ smoke `5876321` and formal `5876591`, where the
   INT8-trained SpinQuant transfer reached 79.6024% versus 78.4098% unrotated;
7. retain accepted FP8-targeted training job `5876984`, then review its
   separately rooted export/load source gate `5881273` before rerunning quality
   and serving gates; do not replace the accepted transfer artifacts.

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

### Isolated min/max versus MSE diagnostic

The accepted quality result uses ordinary min/max weight ranges. To test the
specific hypothesis that paper-style MSE clipping changes the QuaRot versus
unrotated ranking, revision `6f609b0` adds two isolated checkpoints while
preserving the accepted min/max checkpoints as controls. The resulting PPL
matrix is BF16 plus unrotated/QuaRot under min/max and unrotated/QuaRot under
MSE clipping. Group-128, no activation ordering, calibration data, retained
evaluation tokens, activation format, and serving runtime remain unchanged.

Gate job `5875320` completed `0:0`; dependent smoke job `5875322` also
completed `0:0` through `afterok:5875320`. Review accepted the smoke JSON,
logs, observer metadata, 280-linear packed coverage, absence of runtime
`g_idx`, fixed token counts, hashes, and source revision. Its directional PPL
was 4.823038 BF16, 4.985051/5.130191 for unrotated/QuaRot min/max, and
4.993155/5.036449 for unrotated/QuaRot MSE over 4,094 targets per model.

Formal job `5875865` was consequently submitted with
`afterok:5875322` and completed `0:0` over all 331,614 targets. Formal PPL was
5.007820 BF16, 5.136105/5.248356 for unrotated/QuaRot min/max, and
5.163774/5.216687 for unrotated/QuaRot MSE. MSE reduced the QuaRot-minus-
unrotated gap by 52.8622%, but did not reverse the ranking; unrotated min/max
remains the best checkpoint. This accepts the observer diagnostic while
showing that min/max range selection explains only part of the rotation gap.
It does not replace the accepted four-model W4AFP8 result or close the
outstanding serving half of deployment acceptance.

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
load/inference gates. Joint four-model source-gate job `5874345` also completed
from the same revision and its result/capability hashes are frozen in PPL
revision `1f3e4cb`. PPL smoke `5874806` and dependent formal job `5874807`
completed `0:0`; the latter accepted formal PPL of 5.007820 BF16, 5.136105
unrotated, 5.248356 QuaRot-style, and 5.230155 SpinQuant-transfer over 331,614
targets. The deployed-quality gate is complete, but no formal serving speedup
exists. The first SpinQuant checkpoint remains an
INT8-trained rotation transfer diagnostic, not an FP8-optimized endpoint.

BoolQ smoke `5876321` and dependent formal job `5876591` also completed `0:0`.
The transfer checkpoint reached 79.6024%, compared with 78.4098% unrotated,
78.5627% QuaRot-style, and 80.5810% BF16. This motivates the isolated
FP8-targeted training chain, but it does not close the serving half of
deployment acceptance.

The QuaRot-only Gate 1 path remains isolated and does not read the SpinQuant
rotation. Deployed PPL is accepted against the frozen joint gate; serving
remains unsubmitted and must freeze the same provenance before submission.
