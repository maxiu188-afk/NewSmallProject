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

## Accepted matched serving and acceleration

For the accepted INT8-trained SpinQuant-transfer checkpoint set, serving
revision `ad971f9` freezes source gate `5874345` and the existing checkpoint
paths without modifying their producing inputs. Result-gated service smoke
`5882787` completed `0:0` in 8m09s. BF16 and all three W4AFP8 endpoints passed
health, model-listing, and deterministic eight-token completion checks; all
generated texts matched. Ready GPU memory was 34,099 MiB for BF16 and 16,901
MiB for each W4AFP8 checkpoint, and every process recovered to 1--3 MiB after
shutdown. All quantized logs selected `CutlassW4A8LinearKernel`. The smoke JSON
SHA-256 is `d3acdc424cd6796700a9ad937ceb26efca735a9a07adfed1b645a1403d09af7c`.
The optional DeepGEMM import warning was non-fatal and did not replace the
selected CUTLASS path.

Matched benchmark formal `5883004` was submitted only after this review and
completed `0:0` in 19m48s. All eight model/case groups completed 64/64 measured
requests with zero failures, using matched random 256-token inputs, forced
64-token outputs, four warmups, and concurrency 1 or 8. All six quantized
server logs selected `CutlassW4A8LinearKernel`; every server recovered to 1--4
MiB after shutdown. The result SHA-256 is
`df63917675621f891280cf2cf5960e1b394a815f14fd8096125085ea02edae6f`.

| Model | Concurrency | Requests/s | p50 TTFT (ms) | p50 TPOT (ms) | p50 E2E (ms) | Ready GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 1.760 | 21.935 | 8.668 | 568.059 | 34,099 |
| Unrotated W4AFP8 | 1 | 2.506 | 22.006 | 5.986 | 399.145 | 16,803 |
| QuaRot-style W4AFP8 | 1 | 2.504 | 22.611 | 5.983 | 399.448 | 16,803 |
| SpinQuant-transfer W4AFP8 | 1 | 2.506 | 22.208 | 5.981 | 399.154 | 16,803 |
| BF16 | 8 | 11.357 | 104.734 | 9.518 | 703.946 | 34,099 |
| Unrotated W4AFP8 | 8 | 15.807 | 74.066 | 6.873 | 506.352 | 16,803 |
| QuaRot-style W4AFP8 | 8 | 15.919 | 72.081 | 6.850 | 502.254 | 16,805 |
| SpinQuant-transfer W4AFP8 | 8 | 15.777 | 73.719 | 6.864 | 506.332 | 16,804 |

Relative to BF16, the three W4AFP8 variants reduce ready GPU memory by 50.7%,
improve request throughput by 1.42x at concurrency 1 and 1.39--1.40x at
concurrency 8, and reduce p50 end-to-end latency by about 29.7% and 28.1--28.7%
respectively. QuaRot-style and SpinQuant-transfer results remain within about
1% of unrotated W4AFP8 throughput. This establishes the old transfer route's
deployment acceleration, but it does not establish an additional acceleration
effect from rotation itself.

The SpinQuant checkpoint remains an INT8-trained rotation transfer to W4AFP8,
not an FP8-targeted learned endpoint. Its result is retained as the old-rotation
baseline for the separately accepted FP8-targeted endpoint below.

## Accepted transfer-only BoolQ downstream diagnostic

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
rotation. Joint result-gated source job `5881273` completed `0:0` in 54m02s
without a separate smoke or formal dependency. Its Slurm wall-time limit had
been reduced in place from 24 hours to 6 hours; the job was not cancelled or
resubmitted. It writes to a new artifact root and does not overwrite the
accepted INT8-transfer checkpoint.

Review accepted all three export reports, each with 280 packed decoder linears,
group-128 min/max W4 weights, dynamic symmetric per-token FP8 inputs,
`actorder=None`, and zero runtime `g_idx` tensors. The FP8-targeted SpinQuant
report points to training job `5876984`; its observed R1/R2 orthogonality errors
are `4.5449e-7` and `5.9605e-7`. All three quantized fresh-process loads selected
`CutlassW4A8LinearKernel`, and all four variants generated the same eight
greedy tokens. The joint result SHA-256 is
`c292e5ec7abfcfff762e5b1f59139fe0cb423b64cf24ba370969adf222e96a55`;
the capability and source-manifest SHA-256 values are
`495bcc458681f5473e4b1ad50db96b249c82c1ef88cd0f7085e505e5dbdc8d62`
and `df05eae813350c33608acbda2e5c054661cd99d7cfc321d92ee22101b5b96f84`.
This source gate is export/load correctness only. The separately executed PPL
and serving results below complete the core deployment evidence package; they
do not add an FP8-targeted BoolQ result.

### Accepted FP8-targeted deployed PPL

Downstream revision `318bac2` freezes the accepted source artifacts without
changing their paths or hashes. Runnability-only smoke `5884993` completed
`0:0` in 7m00s. Formal job `5884996` then completed `0:0` in 5m18s through
`afterok:5884993`, evaluating 162 x 2048 WikiText-2 sequences and 331,614
scored next-token targets per model.

| Model | Total NLL | Mean NLL | PPL | PPL vs BF16 | PPL vs unrotated |
|---|---:|---:|---:|---:|---:|
| BF16 | 534,230.409917 | 1.611000772 | 5.007820 | baseline | -0.128285 (-2.4963%) |
| Unrotated W4AFP8 | 542,618.332650 | 1.636295008 | 5.136105 | +0.128285 (+2.5617%) | baseline |
| QuaRot-style W4AFP8 | 549,787.787161 | 1.657914886 | 5.248356 | +0.240536 (+4.8032%) | +0.112251 (+2.1855%) |
| FP8-targeted SpinQuant W4AFP8 | 547,964.776632 | 1.652417499 | 5.219583 | +0.211763 (+4.2286%) | +0.083478 (+1.6253%) |

FP8-targeted SpinQuant improves on QuaRot-style by 0.028773 PPL (0.5482%) and
on the old INT8-trained transfer endpoint by 0.010572 PPL (0.2021%). It does
not beat the matched unrotated W4AFP8 control. All three quantized evaluations
record 280 packed decoder linears, group-128 min/max W4 weights,
`actorder=None`, and dynamic symmetric per-token FP8 inputs. The smoke and
formal summary SHA-256 values are respectively
`a5cfc2e28ff23eb0baa7987a3e3f70577441d246c2c7eb8f3fe6d496000c6782`
and `9f17bc86ee664960400dd26f2182ceef72d76dce0a1b75f055331074a7ff7e0a`.

### Accepted FP8-targeted serving and acceleration

Runnability-only service smoke `5884997` completed `0:0` in 8m31s. All four
endpoints returned the same deterministic eight-token text, all three
quantized logs selected `CutlassW4A8LinearKernel`, ready GPU memory was 34,100
MiB for BF16 and 16,901--16,902 MiB for W4AFP8, and every process recovered to
2--3 MiB. Its JSON SHA-256 is
`a99bbf0a324460932623f8d853a107a89202c93ed27c5a244e6944786b45e694`.

Formal benchmark `5884998` completed `0:0` in 18m47s through
`afterok:5884997`. Every one of the eight model/case groups completed 64/64
requests with zero failures, for 512/512 requests overall. The fixed protocol
uses random 256-token inputs, forced 64-token outputs, four warmups,
concurrency 1 or 8, an 8 GiB KV allocation, and BF16 KV dtype.

| Model | Concurrency | Requests/s | p50 TTFT (ms) | p50 TPOT (ms) | p50 E2E (ms) | Ready GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 1.752 | 23.900 | 8.684 | 570.969 | 34,099 |
| Unrotated W4AFP8 | 1 | 2.473 | 25.776 | 6.011 | 404.460 | 16,803 |
| QuaRot-style W4AFP8 | 1 | 2.493 | 22.176 | 6.016 | 401.198 | 16,803 |
| FP8-targeted SpinQuant W4AFP8 | 1 | 2.486 | 23.615 | 6.010 | 402.291 | 16,803 |
| BF16 | 8 | 11.338 | 106.725 | 9.509 | 705.890 | 34,099 |
| Unrotated W4AFP8 | 8 | 15.690 | 75.897 | 6.897 | 508.948 | 16,803 |
| QuaRot-style W4AFP8 | 8 | 15.759 | 73.647 | 6.889 | 507.437 | 16,803 |
| FP8-targeted SpinQuant W4AFP8 | 8 | 15.701 | 74.828 | 6.903 | 509.220 | 16,803 |

For FP8-targeted SpinQuant, W4AFP8 reduces ready GPU memory by 50.7%, improves
request throughput by 1.42x at concurrency 1 and 1.385x at concurrency 8, and
reduces p50 end-to-end latency by 29.5% and 27.9% versus BF16. Relative to
unrotated W4AFP8, throughput changes by only +0.52% and +0.07%. The accepted
acceleration is therefore a W4AFP8 backend result, not evidence that learned
rotation adds serving speed. All six quantized benchmark logs selected the
required CUTLASS kernel, and every server recovered to 1--3 MiB after shutdown.
The formal JSON SHA-256 is
`a10044d965d93ceca756e203a86ad5f72018a2b1028a65b552315a59ea1a80b7`.

Together, formal PPL `5884996` and formal serving `5884998` complete the
FP8-targeted endpoint's core deployment acceptance. FP8 KV was not used or
claimed. Result-gated BoolQ smoke `5886682` was subsequently submitted from
clean revision `183bd8f`, but failed before model execution because the frozen
dataset manifest named the original BoolQ config hash. Corrected revision
`3639cc4` explicitly accepts only that recorded hash while preserving all
dataset revision, fingerprint, example SHA, and row-count checks.

### Accepted FP8-targeted BoolQ downstream diagnostic

Corrected smoke `5886913` completed `0:0` in 4m30s and passed review. Formal
job `5886914` then completed `0:0` in 7m48s through `afterok:5886913`. It
evaluated the same 3,270 examples and 6,540 choice requests per model as the
old-transfer diagnostic.

| Model | Correct | Accuracy | Delta vs unrotated |
|---|---:|---:|---:|
| BF16 | 2,635 / 3,270 | 80.5810% | +2.1713 pp |
| Unrotated W4AFP8 | 2,564 / 3,270 | 78.4098% | baseline |
| QuaRot-style W4AFP8 | 2,569 / 3,270 | 78.5627% | +0.1529 pp |
| FP8-targeted SpinQuant W4AFP8 | 2,625 / 3,270 | 80.2752% | +1.8654 pp |

FP8-targeted SpinQuant gains 61 correct answers over unrotated and 56 over
QuaRot-style. Paired prediction comparisons give new-only/other-only counts of
223/162 versus unrotated (exact McNemar `p=0.00218795`) and 174/118 versus
QuaRot-style (`p=0.00124947`). It remains 10 correct answers and 0.3058 pp
below BF16; the paired 163/173 discordance is not significant (`p=0.623499`).

Compared with the old INT8-trained transfer rotation, the FP8-targeted endpoint
improves BoolQ by 22 correct answers and 0.6728 pp. The paired new-only/old-only
counts are 174/152 (`p=0.244754`), so this old/new improvement is not
statistically significant at the conventional 0.05 level. Its PPL is also only
0.010572 (0.2021%) lower than the old transfer. On serving, the new endpoint is
0.80%/0.48% slower in request throughput and 0.79%/0.57% higher in p50 E2E at
concurrency 1/8, with effectively identical ready memory. These sub-1%
differences are treated as run-to-run equivalence, not a rotation effect;
accepted benchmark `5884998` is sufficient and no repeated new-rotation
acceleration experiment is planned.

The smoke and formal JSON SHA-256 values are respectively
`1aeadb5307a328b15f80eb7377afa4012495b92e3dcd50a347e40cc08e90135a`
and `8e596efea2aeda06d705188060e06db081dd751f610bbef02002c972943f9a4a`.
The formal result records project revision
`3639cc4902297ba1c801824ce4c9e9aaecee9897`, config SHA-256
`78c7b9b578cb3cba116340a8947d73fa77051ebed23d931d24b7fd30cc7f5b15`,
dataset-manifest SHA-256
`66b7a80e9ef1df7d3bd1f07a111824bffcccff56b1ff47c4d9e81afebd1146d5`,
and examples SHA-256
`475e56b71939a8e3db8be48bcc4d344569b36cc660f4086c9ae15858d46a297f`.
This remains downstream quality evidence, not serving-acceleration evidence.
