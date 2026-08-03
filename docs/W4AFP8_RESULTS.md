# W4AFP8 deployed-quality results

## Accepted result

Formal Isambard job `5874807` completed `0:0` on 2026-08-02 after passing
smoke job `5874806` through `afterok`. It evaluated the BF16 reference and all
three packed W4AFP8 checkpoints through vLLM over 162 non-overlapping
2048-token WikiText-2 sequences, scoring 331,614 next-token targets per model.

| Model | Total NLL | Mean NLL | PPL | PPL vs BF16 | PPL vs unrotated |
|---|---:|---:|---:|---:|---:|
| BF16 | 534,230.409917 | 1.611000772 | 5.007820 | baseline | -- |
| Unrotated W4AFP8 | 542,618.332650 | 1.636295008 | 5.136105 | +0.128285 (+2.5617%) | baseline |
| QuaRot-style W4AFP8 | 549,787.787161 | 1.657914886 | 5.248356 | +0.240536 (+4.8032%) | +0.112251 (+2.1855%) |
| SpinQuant-transfer W4AFP8 | 548,635.773268 | 1.654440926 | 5.230155 | +0.222335 (+4.4397%) | +0.094050 (+1.8312%) |

SpinQuant-transfer is 0.018201 PPL, or 0.3468%, lower than QuaRot-style, but
both rotated variants are worse than the matched unrotated W4AFP8 control.
Under this backend-compatible group-128/no-actorder/min-max recipe, neither
offline rotation improves deployed PPL. This finding is specific to the
declared recipe and checkpoint set; it is not a paper-protocol comparison.

## Provenance and acceptance checks

- Source revision: `1f3e4cb2afc81b381b474cd5ebe3c32aa55460bb`.
- Joint four-model source gate: job `5874345`, result SHA-256
  `6a22855a720b95e7230dd15644ced73b28297a960e29d5d2823666de28254f2a`.
- Backend capability SHA-256:
  `048fce61ad390e0ac4dfb89db2f1801c5f47468e993a5ffbdb6b4b4e7d61633a`.
- Retained-token manifest SHA-256:
  `b6ed5f122ba85a9752b3dda699d73405e5a86b7db19050c1f395cbbea450f1cf`.
- Token-ID SHA-256:
  `0f49a76a5cc6f3841356f09fee93eb5a8de9cc37d6b65af54e40614d6eac0de9`.
- Formal result SHA-256:
  `78a81dcdb17d8393247abd65d2f7e00b030e78817dc6d8b0699a8c15e48481f3`.
- The formal JSON identifies checkpoint paths but does not embed their content
  hashes. A post-run audit therefore hashed every file after completion; all
  recorded checkpoint mtimes predate the formal job. Tree SHA-256 values are
  `c38e6f693a466d02e4cdf032250facf22c3f931b86bf6e1663eb89c0b8592496`
  unrotated, `06792e63521d2fa74f0bfa061def342a379453045d6fbc1c034c18446d9a8bf3`
  QuaRot-style, and
  `2037afc16341dcf1d779184fabbfbef0152990ba4132274f717045dce47d8101`
  SpinQuant-transfer. This is a supplemental post-run provenance record, not a
  field produced inside job `5874807`. The supplemental manifest SHA-256 is
  `ce726d8dcc6100de89ee5df679b8b14f3f9b36515ff6d8f7d28714c5785ca8f9`.
- Every W4AFP8 checkpoint contains 280 packed decoder linears, no runtime
  `g_idx`, and compressed-tensors metadata with group size 128,
  `actorder=None`, and `memoryless_minmax` weights.
- The GH200 audit selected `CutlassW4A8LinearKernel` for every required Llama
  shape. The repeated DeepGEMM import warning was an unused optional-backend
  probe; all selected CUTLASS executions and result assertions passed.

Raw smoke/formal JSON and `checkpoint-sha256-postrun.json` remain in the
producing Isambard checkout under `results/vllm-w4afp8-llama2-13b/`, with local
ignored copies retained for review. They are intentionally excluded from Git
by the repository's generated-result policy; the hashes above bind this
reviewed summary to those artifacts.

## Accepted GPTQ observer diagnostic

An isolated 2 x 2 diagnostic was submitted from revision
`6f609b04d1051a60a7d0443796c53d874a9c2144` to test whether the accepted
min/max range selection explains why the unrotated checkpoint beat the
QuaRot-style checkpoint. Gate job `5875320` completed `0:0` in 38m05s and
exported and loaded new unrotated and QuaRot-style checkpoints using
MSE-clipped weight ranges. Smoke job `5875322` then completed `0:0` in 7m35s
through `afterok:5875320`. Both exports contain 280 packed decoder linears,
use the `mse` weight observer with `actorder=None`, and contain no runtime
`g_idx`. Gate and smoke artifacts passed source-revision, observer, retained
token, runtime, and model-set checks.

Formal job `5875865` was submitted with explicit dependency
`afterok:5875322` after manual smoke acceptance and completed `0:0` in 7m02s.
It evaluated all five models over the same 162 x 2048 retained WikiText-2
sequences, scoring 331,614 next-token targets per model.

| Model | Total NLL | Mean NLL | PPL | PPL vs BF16 | MSE vs min/max |
|---|---:|---:|---:|---:|---:|
| BF16 | 534,230.409917 | 1.611000772 | 5.007820 | baseline | -- |
| Unrotated min/max W4AFP8 | 542,618.332650 | 1.636295008 | 5.136105 | +0.128285 (+2.5617%) | baseline |
| QuaRot min/max W4AFP8 | 549,787.787161 | 1.657914886 | 5.248356 | +0.240536 (+4.8032%) | baseline |
| Unrotated MSE W4AFP8 | 544,399.984646 | 1.641667676 | 5.163774 | +0.155953 (+3.1142%) | +0.027669 (+0.5387%) |
| QuaRot MSE W4AFP8 | 547,780.708006 | 1.651862430 | 5.216687 | +0.208866 (+4.1708%) | -0.031670 (-0.6034%) |

MSE clipping improves QuaRot but worsens the unrotated control. The QuaRot
minus unrotated gap falls from 0.112251 PPL (2.1855%) under min/max to
0.052913 PPL (1.0247%) under MSE, a 52.8622% reduction. It does not reverse
the ranking: unrotated min/max remains the best deployed W4AFP8 checkpoint in
this comparison. Observer choice therefore explains about half of the observed
rotation gap, not all of it.

Artifact SHA-256 values are `4d4e938e5fd1cf17e2910d3791d29e35a6bbd56a54be1cabf581ec541c3c8163`
for the unrotated export report,
`629d2a105f4d0e0c924b1a82c60eebcabe209c9cba3afa9a4d9494386380bb0a`
for the QuaRot export report,
`5cf902bdc05e0376247d681ceb2900cadc3525f620d9feac9fd3a8480e991eea`
for the gate result, and
`4058fc424c7dd328b6b9959ec74cf5c5216560c2b43817bc2310c04f67b0e736`
for the smoke result. The formal result SHA-256 is
`c6e0d88b06f27c6147c069db4f7b06e479ab9566348aa28e9cd7953ba97cbbdf`.
A post-run read-only audit confirmed that both checkpoints predate formal
execution. Tree SHA-256 values are
`2e67b7e085d779cbf6d46ee9484be4448985d3a22c3657c82e878754e8c62e1d`
for unrotated MSE and
`81769067eb870faa28e27f8bf29ccb97017f93dc0432b37ab4912f12d3d499a9`
for QuaRot MSE.

The comparison freezes group size 128, block size 128, `actorder=None`,
128 x 2048 calibration tokens, retained PPL tokens, FP8 activation metadata,
and the vLLM runtime. The only intended checkpoint-construction change is the
weight observer: accepted `memoryless_minmax` controls versus MSE clipping with
`maxshrink=0.8`, `grid=100`, and error norm 2.4. This matches the clipping
search range used by the paper code while retaining the deployable group-128
backend contract; it is therefore a deployment-oriented observer diagnostic,
not an exact reproduction of the paper's GPTQ protocol. The formal result is
accepted for this isolated diagnostic only; it does not establish the paper's
group-size -1 GPTQ endpoint or serving acceleration.

## Evidence boundary

This closes the deployed-checkpoint PPL half of the W4AFP8 experiment. It does
not establish acceleration. Matched full-model serving smoke and formal
throughput/latency/memory measurement remain required before any W4AFP8
variant is labelled deployment-complete.

The SpinQuant checkpoint remains an INT8-trained rotation transfer to W4AFP8,
not an FP8-targeted learned endpoint. Its slightly better PPL than QuaRot does
not show that FP8-targeted SpinQuant training is complete.

## Accepted BoolQ downstream diagnostic

Smoke job `5876321` completed `0:0` and was reviewed before formal job
`5876591` was submitted through `afterok:5876321`. The formal run evaluated all
3,270 BoolQ validation examples through vLLM, using the frozen zero-shot prompt,
`no`/`yes` choices, and raw-accuracy construction from the LM Evaluation
Harness commit pinned by QuaRot.

| Model | Correct | Accuracy | Delta vs unrotated |
|---|---:|---:|---:|
| BF16 | 2,635 / 3,270 | 80.5810% | +2.1713 pp |
| Unrotated W4AFP8 | 2,564 / 3,270 | 78.4098% | baseline |
| QuaRot-style W4AFP8 | 2,569 / 3,270 | 78.5627% | +0.1529 pp |
| SpinQuant-transfer W4AFP8 | 2,603 / 3,270 | 79.6024% | +1.1927 pp |

SpinQuant-transfer recovered 39 of the 71 correct-answer gap between the
unrotated checkpoint and BF16. Its paired prediction comparison with unrotated
had 162 SpinQuant-only correct cases and 123 unrotated-only correct cases; the
exploratory exact McNemar p-value was `0.0242202`. This is encouraging evidence
that the learned rotation transfers to the FP8 deployment on BoolQ, despite its
worse deployed WikiText-2 PPL. It does not turn the transfer checkpoint into an
FP8-targeted endpoint and does not establish serving acceleration.

The formal result SHA-256 is
`b278567aae262fdd6f4379d4004817f17faba51fa304077f03dd1479b8bdb824`.
It records 6,540 model requests, project revision
`0f7a2df489f66609c1026ad20987d6ec9f77c7e6`, config SHA-256
`fa44d2c61317022bac346ff4a14824d8da90f192a00c1e6204493ed8d80f818f`,
dataset-manifest SHA-256
`66b7a80e9ef1df7d3bd1f07a111824bffcccff56b1ff47c4d9e81afebd1146d5`,
and example SHA-256
`475e56b71939a8e3db8be48bcc4d344569b36cc660f4086c9ae15858d46a297f`.

This is deliberately labelled a W4AFP8 downstream diagnostic. SpinQuant Table
7 reports LLaMA-2-13B W4A8KV16 BoolQ accuracy of 75.3% for GPTQ and 81.5% for
SpinQuant without online Hadamard transforms; those values are motivation, not
acceptance targets for the current FP8-activation checkpoints.

## FP8-targeted SpinQuant follow-up

The positive BoolQ transfer result motivates a separately named FP8-targeted
rotation-learning run. Its training objective reproduces vLLM's dynamic
per-token FP8 E4M3 input quantizer exactly: one FP32 scale per final-axis row,
maximum magnitude 448, minimum scale `1 / (448 * 512)`, all seven decoder
Linear inputs covered, and `lm_head` excluded. It reuses the accepted
WikiText-2 calibration sequences, initialization, optimizer, and 100-step
schedule so the intended training change is only the activation objective.

Training completion alone is not an accepted W4AFP8 result. The new rotation
must be exported with the frozen group-128/no-actorder/min-max recipe and pass:

1. packed-checkpoint vLLM load/kernel validation;
2. the same formal WikiText-2 PPL and 3,270-example BoolQ protocols; and
3. matched full-model BF16/W4AFP8 serving at concurrency 1 and 8, including
   throughput, TTFT, TPOT, end-to-end latency, GPU memory, request failures,
   recovery, and `CutlassW4A8LinearKernel` log evidence.

One-step training smoke `5876912` completed `0:0` in 2m48s and was manually
accepted. Its result SHA-256 is
`bf1c927f2a821c76aeb7b39d2ef1fbe291bbbf135c33e50af09322631e8a83f2`,
and its rotation tensor SHA-256 is
`a88e745beaf9601d04af0bd07a2c92798bf86062332319782e553fbca9092194`.
Read-only acceptance job `5876983` completed `0:0` in one second. Dependent
100-step formal training job `5876984` then completed `0:0` in 2h10m04s. It
recorded 100 finite losses, 100 non-zero gradient maxima, and training R1/R2
orthogonality errors of `1.7285e-6` and `5.9605e-7` over 800 x 2048 tokens.
The formal result SHA-256 is
`f34f450e03dd59ec9b26942731b470b080908399f9d53089da2fd38bb903f1d1`;
the rotation manifest and SafeTensors SHA-256 values are respectively
`38b423e68f1f503c01a9250a7b850a721182d59ec6ebd4a26077cfdbaa4ba29e`
and `383004941a14e40f256abd4a615246a9adbfe1308ecd08896f167f5b6c2566ec`.

Revision `60aa629` adds a separately configured export/load path for this
rotation. Joint source gate `5881273` was submitted without a smoke or formal
dependency and was `PENDING` at the 2026-08-03 snapshot. It writes to a new
artifact root and does not overwrite the accepted INT8-transfer checkpoint.
PPL, BoolQ, serving, and acceleration remain blocked until this result-gated
source gate completes and its checkpoint, logs, hashes, and kernel evidence
are reviewed.
