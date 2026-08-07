# Current phase status

This file is the concise current-state entry point. The dated update history
through 2026-07-22 is preserved in
[`archive/plans/PHASE_STATUS_2026-07-22.md`](archive/plans/PHASE_STATUS_2026-07-22.md).

## Status summary

| Workstream | Current status | Evidence boundary |
|---|---|---|
| Algorithmic fake quant | Complete for the formal Llama-2-13B study | QuaRot + GPTQ W4A4 reached PPL 5.8376 versus BF16 5.0087; all low-bit rows are floating QDQ and keep K/V at 16 bits |
| SpinQuant fake quant | Corrected-A8 strong-GPTQ diagnostic complete: jobs 5854890/5854891 passed; PPL was 5.0087 BF16, 5.1480 unrotated W4A8, and 5.1627 SpinQuant W4A8 | Group-128 plus activation-order GPTQ is not the paper GPTQ protocol; retain the result under its own label. A group-size -1/no-actorder/clipped paper-GPTQ endpoint is not measured and is outside the closed phase |
| Owned packed W4A8 | Correctness complete on RTX 6000 Ada; selected-linear portability accepted on GH200 | All 280 decoder linears matched the packed oracle on Ada; the Isambard result covers one `q_proj`, not the full decoder; K/V remain BF16 |
| Official QuaRot full model | Complete on RTX 6000 Ada | Real Llama-2-13B W4A4KV4 reduced model-resident memory from 26.29 GB to 7.18 GB but was slower at batch one; no packed-checkpoint PPL result |
| Official QuaRot single block | Complete on RTX 6000 Ada | W4 completed 14/14 cases; 2048-token prefill gained 1.53--1.68x; batch-16/context-4096 layer E2E gained 1.28x; this is not full-model latency |
| vLLM W4A16 serving and quality | Complete on GH200 for matched Llama-2-13B deployed PPL, serving, and layer-0 protocols | Rotated packed W4A16 reached PPL 5.132755 versus 5.289677 unrotated and 5.007820 BF16; W4A16 cut ready GPU memory by 52.7% and improved request throughput by 1.37--1.54x; no downstream-task result |
| SGLang-vLLM W4A16 compatibility | Exact-checkpoint 32-example BoolQ smoke complete on GH200; job `5941763` passed | Both backends scored 64 requests: vLLM 28/32, SGLang 27/32, one prediction disagreement; smoke-only compatibility/score-difference evidence, not formal quality or performance |
| vLLM W4AFP8 deployment | Deployed PPL, matched serving, and BoolQ are accepted on GH200 for both the old INT8-trained SpinQuant transfer and the new FP8-targeted endpoint | FP8-targeted SpinQuant reached PPL 5.219583 and BoolQ 80.2752%; W4AFP8 retained 50.7% lower ready GPU memory plus 1.38--1.42x request throughput versus BF16, while old/new rotation serving differed by less than 1% |

## Current deployment decision

The official-backend route is complete for the current presentation scope.
Its accepted results are consolidated in
[`OFFICIAL_QUAROT_RESULTS.md`](OFFICIAL_QUAROT_RESULTS.md).

The owned W4A8 kernel remains correctness evidence, but its RTX 6000 Ada
performance smoke was much slower than BF16. Stable vLLM does not provide the
selected NVIDIA W4A8 serving path, and this phase will not add an out-of-tree
plugin or another full CUDA backend.

The completed serving route uses offline-fusible QuaRot-style rotation
followed by group-128 GPTQ W4A16 in standard compressed-tensors format. It is
reported as **QuaRot-style W4A16**, not original QuaRot W4A4KV4, because it
omits online MLP/QK transforms and the custom KV4 cache.

This W4A16 decision applies to the completed QuaRot workstream only. The
SpinQuant paper's real-deployment `no_had` route is W4A8, with offline-fused
R1/R2 and activation quantization retained at inference. The completed INT8
W4A8 fake-quant chain uses paper-aligned A8 coverage but a stronger
group-128/activation-order GPTQ protocol, so it is retained as a separately
labelled diagnostic. The paper-GPTQ endpoint is unmeasured and is not a pending
task in this closed phase.

The subsequent joint, hardware-aligned W4AFP8 route for QuaRot and SpinQuant on
GH200 is now complete. Its acceptance pairs deployed-checkpoint PPL with
matched full-model serving performance; fake-quant output is not reused as
W4AFP8 accuracy. Corrected rotation provenance is frozen. The first W4AFP8
gates were cancelled before allocation when the GPTQ protocol changed. Jobs
`5859043`/`5859044` later failed stale-calibration validation, and
`5873446`/`5873447` passed calibration and backend audit before an over-strict
observer-name check rejected LLM Compressor's canonical `memoryless_minmax`
name. Revision `9d691e9` accepts only the equivalent
`minmax`/`memoryless_minmax` names and still rejects MSE observers. QuaRot
validation job `5873544` completed in 38m30s, and SpinQuant-transfer job
`5873545` then completed in 19m44s through `afterok:5873544`. Both returned
`0:0`; all fixed-token outputs matched BF16. Joint four-model source-gate job
`5874345` then completed `0:0` in 4m35s. Its inference and capability SHA-256
hashes are frozen in PPL revision `1f3e4cb`, whose provenance and retained-token
preflights passed. PPL smoke `5874806` completed `0:0` in 6m24s; formal job
`5874807` then completed `0:0` in 5m00s through `afterok:5874806`. Formal PPL
was 5.007820 BF16, 5.136105 unrotated W4AFP8, 5.248356 QuaRot-style W4AFP8,
and 5.230155 SpinQuant-transfer W4AFP8. Both rotations were worse than the
matched unrotated control; SpinQuant-transfer was slightly better than
QuaRot-style. See [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md). The deployed-quality
result is paired with accepted matched serving benchmark `5883004`.
See [`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md) and
[`W4AFP8_ISAMBARD_RUNBOOK.md`](W4AFP8_ISAMBARD_RUNBOOK.md).

To isolate why the backend-compatible unrotated GPTQ checkpoint outperformed
the QuaRot-style checkpoint, revision `6f609b0` adds a matched MSE-clipping
diagnostic without replacing the accepted min/max artifacts. Gate `5875320`
and smoke `5875322` completed `0:0`, with smoke constrained by
`afterok:5875320`. Review accepted the source revision, observer metadata,
packed coverage, absence of runtime `g_idx`, fixed token counts, and artifact
hashes. Formal job `5875865` was then submitted through
`afterok:5875322` and completed `0:0` in 7m02s. Over 331,614 targets, MSE
PPL was 5.163774 unrotated and 5.216687 QuaRot-style. The matched min/max PPL
was 5.136105 and 5.248356, so MSE reduced the rotation gap by 52.8622% without
changing the winner. The accepted four-model deployment result above remains
unchanged.

The W4AFP8 **formal deployment result is a two-part evidence package**, not a
performance-only benchmark. Accuracy must be measured from the packed
checkpoints through vLLM on the retained 162 x 2048 WikiText-2 tokens
(331,614 scored next-token targets), reporting total NLL and PPL for BF16,
unrotated W4AFP8, QuaRot-style W4AFP8, and SpinQuant W4AFP8. Acceleration must
be measured separately with the matched serving protocol, reporting throughput,
TTFT, TPOT, end-to-end latency, GPU memory, request completion, and selected
kernel evidence. A method is not deployment-complete if either formal accuracy
or formal serving evidence is missing.

## Latest accepted result

FP8-targeted SpinQuant formal training job `5876984` completed `0:0` in
2h10m04s and is accepted as rotation-training evidence. It recorded 100 finite
losses and 100 non-zero gradient maxima over 800 x 2048 calibration tokens.
Training R1/R2 orthogonality errors were `1.7285e-6` and `5.9605e-7`; the
rotation SafeTensors SHA-256 is
`383004941a14e40f256abd4a615246a9adbfe1308ecd08896f167f5b6c2566ec`.
This does not by itself establish W4AFP8 quality or acceleration. Isolated
source gate `5881273` completed `0:0` in 54m02s from clean revision `60aa629`
without replacing the accepted INT8-transfer artifacts. It is accepted for
export/load correctness: all three W4AFP8 checkpoints contain 280 packed
decoder linears, use group-128 min/max W4 with dynamic per-token FP8 inputs,
have no runtime `g_idx`, load through the CUTLASS W4AFP8 kernel, and reproduce
the same eight greedy tokens as BF16. The joint result SHA-256 is
`c292e5ec7abfcfff762e5b1f59139fe0cb423b64cf24ba370969adf222e96a55`.
Its wall-time limit had been reduced in place from 24 hours to 6 hours without
cancellation or resubmission.

Revision `318bac2` freezes that accepted source gate into separate downstream
PPL and serving configurations. Runnability-only PPL smoke `5884993` completed
`0:0` in 7m00s, and formal PPL `5884996` completed `0:0` in 5m18s through
`afterok:5884993`. Over 331,614 scored targets, PPL was 5.007820 BF16,
5.136105 unrotated W4AFP8, 5.248356 QuaRot-style W4AFP8, and 5.219583
FP8-targeted SpinQuant W4AFP8. The new SpinQuant endpoint improves on QuaRot
by 0.028773 PPL and on the old transfer endpoint by 0.010572 PPL, but remains
0.083478 PPL (1.6253%) worse than the matched unrotated control. The formal
PPL JSON SHA-256 is
`9f17bc86ee664960400dd26f2182ceef72d76dce0a1b75f055331074a7ff7e0a`.

Runnability-only service smoke `5884997` completed `0:0` in 8m31s, and formal
serving benchmark `5884998` completed `0:0` in 18m47s through
`afterok:5884997`. All eight model/concurrency groups completed 64/64 requests
with zero failures; all six quantized logs selected
`CutlassW4A8LinearKernel`, and every server recovered to 1--3 MiB. For the
FP8-targeted SpinQuant endpoint, request throughput was 1.42x BF16 at
concurrency 1 and 1.385x at concurrency 8, while ready GPU memory fell from
34,099 MiB to 16,803 MiB (50.7%). Its throughput differed from unrotated
W4AFP8 by only +0.52% and +0.07%, so acceleration is attributed to W4AFP8
deployment rather than rotation. The formal serving JSON SHA-256 is
`a10044d965d93ceca756e203a86ad5f72018a2b1028a65b552315a59ea1a80b7`.
The serving comparison retains BF16 KV cache; FP8 KV remains outside this
accepted result. Separately rooted BoolQ revision `183bd8f` passed source/data
preflight, but initial smoke `5886682` failed `1:0` in five seconds before model
execution because the immutable dataset manifest recorded the original BoolQ
config hash. Revision `3639cc4` fixes this fail-closed by explicitly freezing
that original hash while retaining the same revision, fingerprint, examples
SHA, and row-count checks. Corrected result-gated smoke `5886913` completed
`0:0` in 4m30s. Dependent formal job `5886914` then completed `0:0` in 7m48s
through `afterok:5886913`, scoring all 3,270 validation examples: 80.5810%
BF16, 78.4098% unrotated, 78.5627% QuaRot-style, and 80.2752% FP8-targeted
SpinQuant. The new endpoint is +1.8654 pp over unrotated, +1.7125 pp over
QuaRot-style, and -0.3058 pp from BF16. Its formal JSON SHA-256 is
`8e596efea2aeda06d705188060e06db081dd751f610bbef02002c972943f9a4a`.
Against the old transfer rotation it gains 0.6728 pp and 22 correct answers,
but the paired new-only/old-only counts of 174/152 give an exploratory exact
McNemar p-value of `0.244754`. The accepted old/new serving results differ by
less than 1% in throughput and end-to-end latency with effectively identical
ready memory;
no additional FP8-targeted serving run is needed or planned.

The previously missing serving experiment for the accepted INT8-trained
rotation transfer is complete. Result-gated service smoke `5882787` completed
`0:0` in 8m09s and is accepted: all four endpoints succeeded, all generated
texts matched, GPU memory recovered after shutdown, and all three quantized
logs selected `CutlassW4A8LinearKernel`. Its result SHA-256 is
`d3acdc424cd6796700a9ad937ceb26efca735a9a07adfed1b645a1403d09af7c`.
Benchmark formal `5883004` was then submitted from the same frozen revision
`ad971f9` and completed `0:0` in 19m48s. All eight model/concurrency groups
completed 64/64 requests with zero failures. The three W4AFP8 checkpoints cut
ready GPU memory from 34,099 MiB to 16,803--16,805 MiB and improved request
throughput by 1.42x at concurrency 1 and 1.39--1.40x at concurrency 8 versus
BF16. QuaRot and SpinQuant were effectively tied with unrotated W4AFP8, so the
accepted acceleration is attributed to W4AFP8 deployment, not rotation. The
formal result SHA-256 is
`df63917675621f891280cf2cf5960e1b394a815f14fd8096125085ea02edae6f`.

The earlier transfer-only BoolQ diagnostic is formal job `5876591`, submitted
after manual acceptance of smoke `5876321`. It scored all 3,270 validation examples:
80.5810% BF16, 78.4098% unrotated W4AFP8, 78.5627% QuaRot-style W4AFP8, and
79.6024% SpinQuant-transfer W4AFP8. The formal JSON SHA-256 is
`b278567aae262fdd6f4379d4004817f17faba51fa304077f03dd1479b8bdb824`.
This is downstream quality evidence, not serving-acceleration evidence.

The accepted min/max-versus-MSE job `5875865` remains the latest observer
diagnostic.
It scored 331,614 retained WikiText-2 targets per model and confirmed that MSE
clipping improves QuaRot PPL from 5.248356 to 5.216687 while worsening the
unrotated control from 5.136105 to 5.163774. Its formal result SHA-256 is
`c6e0d88b06f27c6147c069db4f7b06e479ab9566348aa28e9cd7953ba97cbbdf`.
The primary four-model deployed-quality result remains job `5874807`, including
SpinQuant-transfer; neither result is serving-acceleration evidence.

The Isambard evidence chain is complete through Llama-2-13B offline inference,
OpenAI-compatible service smoke, matched full-model serving, and the dependent
real-vLLM layer-0 diagnostic. Formal serving job `5780631` completed all six
model/concurrency groups with 64/64 requests and zero failures. Against BF16,
both W4A16 variants reduced ready GPU memory from 34,099 to 16,125 MiB, raised
request throughput by 1.53--1.54x at concurrency 1 and 1.37x at concurrency 8,
and reduced p50 E2E by 26.9--35.2%.

Dependent jobs `5784966` and `5784967` then passed the packed-W4 hook smoke and
eight-case single-block comparison. W4A16 reduced layer-0 parameter bytes by
74.2%, but measured speedups were shape-dependent and small: approximately
0.96--1.05x versus BF16. This diagnostic used eager execution and is not a
replacement for the compiled serving result.

The deployed-checkpoint quality gate has also passed. Smoke job `5839415` and
formal job `5839419` completed with exit code `0:0`; the formal run scored
331,614 WikiText-2 targets through the real vLLM checkpoints. PPL was 5.007820
for BF16, 5.289677 for unrotated W4A16, and 5.132755 for rotated W4A16. Rotation
reduced PPL by 2.97% relative to unrotated W4A16 and recovered 55.67% of its PPL
gap to BF16 while leaving serving performance materially unchanged.
Downstream-task or broader generation evaluation is not covered and is not
planned for this closed phase; deployed-checkpoint perplexity is complete.

The SpinQuant work is isolated under its own implementation, configuration,
script, and test directories. It does not modify the completed QuaRot pipeline
or the remote vLLM checkout. Jobs `5841874` and `5842047` accepted one-step and
100-step full-model rotation training; dependent jobs `5847440` and `5847441`
then accepted the matched held-out fake-quant PPL gate. Learned offline R1/R2
reached PPL 5.0937, compared with 5.1763 unrotated and 5.7078 for the fixed
random-Hadamard R1/R2 mechanism baseline. This W4A16 result is retained as a
weight-only ablation, not the paper deployment target.

The first activation-only rotation chain was superseded after the A8 audit
found a paper-protocol mismatch at `o_proj`. Corrected jobs `5854268` and
`5854269` passed the one-step and 100-step W16A8 learning gates. Corrected GPTQ
W4A8 PPL smoke `5854890` and formal job `5854891` then passed on the retained
held-out tokens. SpinQuant no-had reached PPL `5.1627`, slightly worse than the
matched unrotated W4A8 control at `5.1480`; this completes the reproduction
chain but does not establish a learned-rotation benefit. See
[`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md) for protocol and claim boundaries
and [`SPINQUANT_PLAN.md`](SPINQUANT_PLAN.md) for the retained execution record.

See [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md) for the accepted metrics,
[`VLLM_W4A16_ISAMBARD_RUNBOOK.md`](VLLM_W4A16_ISAMBARD_RUNBOOK.md) for the
procedure, and
[`QUAROT_REAL_DEPLOYMENT_ROADMAP.md`](QUAROT_REAL_DEPLOYMENT_ROADMAP.md) for
the full decision record, and [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) for the
claim-to-artifact closeout index.

## Future separately scoped comparison

The current QuaRot/SpinQuant evidence phase is closed, but this does not close
serving-system research. A future study may compare SGLang and vLLM under the
same GH200, checkpoint, request stream, quality gate, and observed resource
budget. It is defined separately in
[`SERVING_BACKEND_COMPARISON_PLAN.md`](SERVING_BACKEND_COMPARISON_PLAN.md).
Its read-only compatibility audit and result-gated smoke are complete, but no
SGLang performance result is currently accepted. The source audit predicted
that SGLang `0.5.16` could not dispatch the unchanged dense compressed-tensors
W4AFP8 checkpoint; job `5896201` now confirms that negative runtime boundary.
Result-gated smoke job `5895081` was submitted from clean revision `3578427`
without a formal-job dependency. It completed, but its SGLang cases used the
Cray base interpreter after an erroneous virtual-environment symlink resolution
and failed before model loading. The vLLM controls passed; the SGLang outcomes
are invalid as compatibility evidence and require one corrected replacement
smoke. Corrected job `5895358` is submitted from clean revision `5355df8`
without a formal-job dependency. It completed, but both SGLang cases stopped
before model loading because optional `sgl-deep-gemm` asserted that `CUDA_HOME`
was unset. This second environment failure is also invalid as model
compatibility evidence. A login-node import check passed after binding the
environment toolkit and disabling unused JIT DeepGEMM; one final corrected
compatibility smoke was therefore required before any formal comparison
decision. Final corrected job `5896201`, submitted from clean revision
`f33d002` without a formal-job dependency, completed `0:0`. Both vLLM controls
passed, including CUTLASS W4AFP8 selection. SGLang reached the exact W4AFP8
loader and raised `No compressed-tensors compatible scheme was found`, so the
unchanged-checkpoint W4AFP8 comparison stops. The SGLang BF16 weights loaded,
but its default FA3 backend failed because the installed `sgl_kernel` lacks
`flash_ops`; this is not BF16 incompatibility. One BF16-only corrected smoke is
required before Track A can be considered, and no formal job is submitted. See
[`SERVING_BACKEND_COMPARISON_RESULTS.md`](SERVING_BACKEND_COMPARISON_RESULTS.md).

The next authorized step is Track C first: exact accepted QuaRot-style rotated
W4A16 on the frozen zero-shot BoolQ protocol. Its immutable-input audit binds
the checkpoint tree SHA-256
`2f22f56a5edb32e037416c78be49e617bcee796abca26822704a6ef825ff8e99`,
280 packed decoder linears, group-128 static-actorder W4A16 metadata, and the
accepted source result. Job `5905638` is not accepted: vLLM completed 32
examples/64 requests with 28 correct, while SGLang never reached model loading
because the runner resolved its virtual-environment Python symlink to the Cray
base interpreter. This is a harness failure, not SGLang W4A16 incompatibility;
the vLLM result remains smoke-only.

The executable-path regression was fixed and tested in revision `a3a23a7`.
Replacement `5913876` then reached SGLang's `CompressedTensorsWNA16` path but
recorded an incomplete comparison because the GPTQ-to-Marlin repack JIT could
not find the already installed Ninja `1.13.0` executable. vLLM again completed
32 examples/64 requests with 28 correct. The failure is environment evidence,
not end-to-end SGLang compatibility or incompatibility evidence.

Revision `5445e73` adds the SGLang environment `bin` directory to `PATH`,
preflights Ninja, and requires both backend cases to pass before the batch job
can return success. Replacement `5917675` correctly failed closed (`1:0`) with
an incomplete comparison. vLLM again scored 28/32. SGLang reached
`CompressedTensorsWNA16` GPTQ-to-Marlin JIT compilation, where NVCC used the
default GCC `7.5.0` host toolchain and failed because the C++20 `<version>`
header was unavailable. GCC 14 is installed and passes the corresponding
read-only header check, so this remains an environment binding failure rather
than W4A16 compatibility or incompatibility evidence.

Revision `f083519` implements the bounded correction without changing the
model, 32-example BoolQ subset, quantization, or one-GH200/two-hour resource
request. It binds GCC/G++ `14.3.0`, Ninja `1.13.0`, NVCC `13.3.73`, SGLang
`0.5.16`, PyTorch `2.11.0+cu129`, the remaining WNA16 dependency versions,
and SHA-256 values for the loaded WNA16/log-probability/JIT sources including
the exact Marlin `.cuh`. It also corrects SGLang's `logprob_start_len` offset
so the first BoolQ continuation token is scored. The result-gated batch now
compiles and executes the exact GPTQ-to-Marlin JIT before either backend is
evaluated.

Login-node acceptance passed 18 focused tests, the complete version and source
hash gates, and the immutable checkpoint/source-result validation. Replacement
smoke `5927118` was submitted alone from clean revision `f083519`, then failed
closed after 51 seconds during runtime preflight. NVCC compiled the exact
Marlin CUDA source with GCC `14.3.0`, but TVM-FFI expected
`${CUDA_HOME}/lib64/libcudart.so` while the pip CUDA `13.3` layout provided only
`${CUDA_HOME}/lib/libcudart.so.13`; linking failed before model loading. This is
environment-layout evidence, not SGLang W4A16 compatibility evidence.

Revision `75a805f` adds and validates the exact `libcudart.so.13` compatibility
link, exports its runtime directory through `LD_LIBRARY_PATH`, and performs a
real CUDART link and dynamic-load probe before the exact Marlin JIT. The helper
passed twice on the existing environment, the probe returned runtime version
`13000`, 19 focused tests passed on Isambard, and the immutable-input gate
passed again. Replacement smoke `5932590` was submitted alone from clean
revision `75a805f` after `sbatch --test-only` accepted the unchanged request.
It failed closed (`1:0`) after 27 seconds before model loading. CUDART linking,
dynamic loading, and the exact Marlin JIT build all passed, but its first
synthetic CUDA call failed with `CUDA driver version is insufficient for CUDA
runtime version`. The JIT was compiled and linked against pip CUDA 13.3, so
this verifies the prior layout fix while exposing a separate CUDA 13/driver
boundary; it is not checkpoint compatibility evidence.

Revision `bfe2e3a` moves the SGLang source JIT to the complete system
`cuda/12.6` toolkit (`nvcc 12.6.77`, `libcudart.so.12`) and
`gcc-native/13.2`, while retaining the pinned PyTorch cu129 environment. It
removes the artificial pip-CUDA-13 link and uses the isolated TVM-FFI cache
namespace `cuda-12.6-gcc-13.2-tvmffi-0.1.11` so the CUDA 13 shared object
cannot be reused. Eighteen focused tests, the immutable-input gate, and
`sbatch --test-only` passed on Isambard. The exact generated Marlin source also
compiled for `sm_90a`, linked to system `libcudart.so.12`, and dynamically
loaded on the login node; GPU execution remains deliberately unclaimed.
Replacement smoke `5940088` was submitted alone from clean revision `bfe2e3a`
with no formal dependency. It failed closed (`1:0`) after 8 minutes 15 seconds,
but the CUDA 12.6 preflight and exact Marlin synthetic GPU execution passed.
SGLang loaded the unchanged checkpoint as `compressed-tensors` W4A16, used
6.82 GB for weights, reached healthy serving, and returned HTTP 200 for the
first eight-request BoolQ batch. This establishes unchanged-checkpoint
load/serve compatibility. vLLM again completed all 64 requests with 28/32
correct, but score parity remains unverified.

The incomplete comparison was caused by the harness parser, not the backend.
SGLang `0.5.16` returns an unscored `[None, token_id]` entry at
`logprob_start_len`, followed by scored tokens. The request must retain
`continuation_start - 1` to score the first (single-token here) continuation;
the parser had instead required the response length to equal only the
continuation length. Revision `b348fc3` now strictly validates and removes the
sentinel before checking token order and summing finite continuation scores.
It also binds the exact `logprob_result_processor.py` source hash. Twenty
focused tests, all seven source hashes, the immutable-input gate, and
`sbatch --test-only` passed on Isambard. Replacement smoke `5941763` was
submitted alone from clean revision `b348fc3` with no formal dependency. It
completed `0:0` in 3 minutes 33 seconds with empty stderr. The exact Marlin GPU
preflight and all seven source-hash gates passed; both backends scored all 64
choice requests. vLLM reached 28/32 (87.5%) and SGLang 27/32 (84.375%). They
disagreed only on example 6; the mean absolute choice-loglikelihood difference
was 0.0351943 and the maximum was 0.0953803. All manifest entries and retained
artifact/log hashes were recomputed successfully. This accepts the bounded
exact-checkpoint compatibility and smoke score-difference result, not formal
BoolQ quality or serving performance. No formal job is submitted.

The next bounded step was submitted but is not accepted. Revision `78ed96c`
adds one repository-owned OpenAI streaming client and a pre-hashed 64-request
corpus that reconstructs the accepted random 256-token workload. It explicitly
aligns BF16 KV, the 8 GiB KV budget, prefix-cache disabling, and chunked-prefill
disabling across both backends. Result-gated job `5944823` was submitted alone
with only concurrency 1 and eight measured 64-token outputs per backend. It
failed closed `1:0` after 4 minutes 12 seconds for two harness reasons. SGLang
passed the CUDA 12.6 Marlin preflight, loaded W4A16, allocated the exact 8 GiB
BF16 KV pool, and reached healthy service; the launcher then omitted its served
model name, so the harness rejected the checkpoint path returned by
`/v1/models` before measured requests. vLLM completed all 13 requests but was
rejected by an invalid old generated-output equality gate. The old serving
artifact retained no prompts, so that comparison is diagnostic rather than
request identity. A harness-only replacement is required. No SGLang
performance number is accepted, and no concurrency-8 or formal paired-
repetition job is submitted. Revision `ac82347` fixes only those two harness
gates. Replacement result-gated smoke `5949509` was submitted alone from that
clean revision with no formal dependency. Its result is unreviewed; submission
is not evidence that either backend passed, and it will not be monitored until
the user explicitly requests inspection.
