# SpinQuant independent reproduction plan

## Scope and implementation policy

SpinQuant is implemented inside this repository but remains isolated from the
existing QuaRot pipeline:

- implementation: `repro/spinquant/`;
- runnable entry points: `scripts/spinquant/`;
- pinned experiment definitions: `configs/spinquant/`;
- tests: `tests/spinquant/`;
- generated artifacts: `results/spinquant/`, ignored by Git.

The locally retained `SpinQuant/` checkout is an untracked, read-only reference.
It is not imported, packaged, or copied. The implementation is derived from the
paper's equations and tensor placement, then expressed against the current
project's PyTorch and fake-quant primitives. This avoids coupling the
reproduction to the reference repository's pinned Transformers 4.44-era
modified Llama forward, optimizer, monkey patches, and serialized rotations.

## Accepted local mechanism smoke

The first phase implements:

- a global trainable residual rotation R1;
- one independently trainable head-wise R2 per layer;
- reproducible random signed-Hadamard initialization;
- a Cayley update derived from the paper's Stiefel-manifold equations;
- an exact linear-solve path for numerical tests;
- a fixed-point path for eventual model-sized rotation matrices;
- W4 floating QDQ with a straight-through gradient;
- the general, non-symmetric R2 offline V/O fusion convention;
- a training-only adapter around standard Transformers Llama leaf modules;
- GQA-aware R2 blocks and tied-embedding support without copying model source;
- group-wise W4 STE-QDQ with all pretrained model parameters frozen;
- power-of-two and structured-Hadamard initialization, including the
  `40 x 2^k` family needed by Llama-2-13B's residual width;
- SafeTensors rotation storage plus a checksummed JSON provenance manifest;
- R2 V/O transforms implemented as per-head batched matrix products rather
  than model-width block-diagonal temporary tensors.

The deterministic CPU smoke uses two synthetic 8-wide V/O pairs and 40
linearly decayed optimization steps. It reduced W4 fake-quant MSE from
`4.726889` to `1.882067` (`0.3982x`). Maximum orthogonality error was
`2.20e-7` for R1 and `4.17e-7` for R2.

This is mechanism evidence only. It is not evidence for a pretrained model,
WikiText-2 perplexity, GPTQ, downstream accuracy, a packed checkpoint, a
low-bit kernel, or deployment performance.

The second local smoke installs the independent adapter on a two-layer,
random-config Transformers Llama with four Q heads, two KV heads, and tied
embeddings. A W16 check preserves logits and all hidden states after restoring
the learned residual basis. The W4A16 training check uses 48 synthetic tokens
and 30 Cayley steps; logit MSE fell from `3.14883e-5` to `3.04078e-5`
(`0.9657x`). Only R1/R2 received gradients. Maximum orthogonality error was
`2.38e-7` for R1 and `1.64e-7` for R2.

Rotation artifacts contain only `r1` and `r2` in
`rotations.safetensors`. `rotation-manifest.json` records tensor shapes,
orthogonality error, provenance, and the SafeTensor SHA256. Loading rejects
unexpected keys, shapes, dtypes, excessive orthogonality error, and checksum
changes. Pickle and `torch.load` are not used.

The 800-sample Llama-2-13B rotation-calibration configuration is now pinned to
the existing model and WikiText-2 revisions. The preparation script joins
WikiText-2 train rows with two newlines, tokenizes once without special tokens,
and samples 800 length-2048 windows with replacement from a local seeded RNG.
The resulting token JSON and manifest are independently checksummed.

This transparent protocol follows the sample count and sequence length declared
by the paper, but it is not claimed to reproduce Meta's unreleased internal
data loader or the released reference repository's row-chunking behavior.
Those protocol variants must remain separately labelled if both are evaluated.
The 800-window artifact has not been materialized in the current local
environment.

The matching training entry point is also configuration-driven. It requires
exactly `steps x gradient_accumulation_steps` sequences rather than silently
cycling data. The current Llama-2-13B W4A16 configuration freezes 100 optimizer
steps, eight microbatches per update, learning rate 1.5 with linear decay,
fixed-point Cayley updates, group-128 symmetric weight QDQ, BF16 base weights,
and activation/KV precision at 16 bits. The runner rejects CPU execution,
unpinned model snapshots, calibration shape drift, architecture drift, and
zero/non-finite rotation gradients.

The paired smoke configurations retain sequence length 2048 and all model,
quantization, runtime, and optimizer settings; they reduce only the calibration
set from 800 to 8 sequences and the optimizer updates from 100 to 1. The
same-revision smoke and formal CUDA runs have now passed on Isambard GH200.

The Isambard one-step entry point is
`scripts/run_isambard_spinquant_llama2_13b_1step_smoke.sbatch`. It uses the
existing `newsmallproject-llmcompressor-0.12.0` environment read-only because
that environment already provides the pinned GH200 PyTorch, Transformers,
Datasets, Accelerate, and SafeTensors versions. It does not install into or
execute through the accepted vLLM environment. Generated calibration and
rotation artifacts live under a separate
`${PROJECTDIR}/${USER}/newsmallproject-spinquant/` root.

The smoke job requires a clean checkout, exports a writable job-specific temporary
directory before importing PyTorch, runs with Hugging Face and Datasets in
offline mode, hashes every controlling source/config file, and refuses partial
or overwritten artifacts. Calibration reuse additionally requires the exact
project revision, calibration-config hash, model-config hash, pinned model
definition, and pinned dataset definition. The job has been prepared and
locally validated.

The one-step GH200 smoke passed as Isambard job `5841874` in 3 minutes 47
seconds with exit code `0:0`. It consumed eight 2048-token calibration
sequences, produced a non-zero maximum rotation gradient of `0.285095`, and
retained maximum R1/R2 orthogonality errors of `1.31e-6` and `5.96e-7` during
training. The accepted 102.5 MiB SafeTensors artifact has SHA256
`48e3d10e0a62dd6c78d4e5ff507ec387105c5f11771aa71fc70a4fb6588f9abb`.
This establishes full-model execution only, not learned-rotation quality.

The formal training entry point is
`scripts/run_isambard_spinquant_llama2_13b_100step.sbatch`. It requests six
hours based on the observed one-step runtime and validates the completed smoke
result plus every smoke controlling-source hash before training. Success
requires all 100 loss and non-zero gradient records, bounded orthogonality,
the expected R1/R2 shapes, and a checksummed SafeTensors artifact.

The formal training passed as Isambard job `5842047` in 2 hours 17 minutes 51
seconds with exit code `0:0`. All 100 gradient maxima were finite and non-zero,
the first/last ten-step mean losses were `1.73751` and `1.57174`, and the best
update loss was `1.34794` at step 94. The final R1/R2 training orthogonality
errors were `1.34e-6` and `5.96e-7`. The accepted rotation SafeTensors SHA256
is `303c614f425ea5d37e138643dd52a2410a4747fda7e3038587c93a2db05a57fe`.
This training artifact subsequently passed the matched held-out evaluation.

The accepted quality gate uses
`scripts/run_isambard_spinquant_llama2_13b_fake_quant_ppl.sbatch`. It compares
BF16, unrotated W4A16, fixed QuaRot-style R1/R2 W4A16, and learned SpinQuant
R1/R2 W4A16. All four cases use the same pinned Llama-2-13B snapshot and the
same retained 162 x 2048 WikiText-2 test token artifact as the deployed QuaRot
PPL study. R1/R2 are fused into standard Llama parameters before one-time
group-128 W4 floating QDQ; no online rotation wrapper, GPTQ, packing, integer
kernel, or deployment backend is involved.

The reduced smoke job `5847440` completed with exit code `0:0` in 2 minutes 29
seconds. The dependent formal job `5847441` completed with exit code `0:0` in
3 minutes 25 seconds and evaluated 162 sequences with 331,614 scored tokens.
The matched PPL results were BF16 `5.0087`, unrotated W4A16 `5.1763`, fixed
random-Hadamard R1/R2 W4A16 `5.7078`, and learned SpinQuant R1/R2 W4A16
`5.0937`. The learned rotations reduced PPL by 1.60% relative to unrotated W4
and by 10.76% relative to the fixed R1/R2 mechanism baseline. See
[`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md) for hashes and evidence
boundaries.

Run it with:

```bash
python scripts/spinquant/run_tiny_fake_quant.py \
  --config configs/spinquant/tiny_w4a16_fake_quant_smoke.json \
  --output results/spinquant/tiny-w4a16-smoke.json

python scripts/spinquant/run_tiny_llama_fake_quant.py \
  --config configs/spinquant/tiny_llama_w4a16_fake_quant_smoke.json \
  --output results/spinquant/tiny-llama-w4a16-smoke.json

python scripts/spinquant/prepare_wikitext2_calibration.py \
  --config configs/spinquant/llama2_13b_rotation_calibration_800.json \
  --model-snapshot /path/to/pinned/model/snapshot \
  --output-dir results/spinquant/llama2-13b-rotation-calibration

python scripts/spinquant/train_llama_rotations.py \
  --config configs/spinquant/llama2_13b_w4a16_rotation_train_100.json \
  --model-snapshot /path/to/pinned/model/snapshot \
  --calibration-manifest \
    results/spinquant/llama2-13b-rotation-calibration/calibration-manifest.json \
  --output-dir results/spinquant/llama2-13b-w4a16-rotations

sbatch --test-only \
  scripts/run_isambard_spinquant_llama2_13b_1step_smoke.sbatch
sbatch scripts/run_isambard_spinquant_llama2_13b_1step_smoke.sbatch

sbatch --test-only \
  scripts/run_isambard_spinquant_llama2_13b_100step.sbatch
sbatch scripts/run_isambard_spinquant_llama2_13b_100step.sbatch

sbatch \
  scripts/run_isambard_spinquant_llama2_13b_fake_quant_ppl.sbatch smoke
sbatch --dependency=afterok:<smoke-job> \
  scripts/run_isambard_spinquant_llama2_13b_fake_quant_ppl.sbatch formal
```

## Paper-aligned fake-quant target and deployment decision

The paper-aligned SpinQuant quality target is corrected from W4A16 to the
paper's `SpinQuant_no_had` W4A8 route. The current fake-quant gate uses
W4A8KV16:

- learn R1/R2 with W16 weights and dynamic per-token asymmetric A8 QDQ;
- fuse R1/R2 into ordinary Llama weights offline;
- apply group-128 GPTQ W4 after rotation learning;
- retain A8 execution at inference, with KV left at 16 bits initially;
- do not add online Hadamard R3/R4 to this no-had deployment path.

After this fake-quant gate is accepted, the GH200 real-deployment target is
W4AFP8 rather than INT8 W4A8. That deployment is a hardware-aligned extension,
must be evaluated independently, and includes both QuaRot and SpinQuant. See
[`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md).

The completed W4A16 SpinQuant experiment is retained as a matched weight-only
ablation. The completed QuaRot-style W4A16 vLLM deployment remains valid for
its own workstream; neither result is relabelled as paper-faithful SpinQuant
deployment.

### Migration difficulty assessment

| Area | Difficulty | Assessment |
|---|---|---|
| Fake-quant algorithm and evaluation | Low to medium | Offline R1/R2 fusion, A8 QDQ, GPTQ primitives, and the formal learned rotations already exist. The main work is composing post-learning GPTQ with the W4A8 evaluation path and adding provenance/quality gates; rotation retraining is not required. |
| W4AFP8 compressed-checkpoint export | Medium | The existing W4A16 exporter must emit group-128 W4 plus dynamic per-token FP8 metadata, omit unsupported runtime `g_idx`, and validate loader coverage and runtime form. |
| GH200 vLLM W4AFP8 correctness smoke | Medium | Fresh-process loading, fixed-token output, selected-kernel/runtime-form, FP8-scale, and packed-linear coverage gates are required before quality or timing. |
| GH200 accelerated W4AFP8 serving | Medium / gated | The installed Hopper path appears promising, but acceleration is not accepted until logs prove the intended W4AFP8 kernel and the matched full-model benchmark beats the BF16 reference. |
| FP8-targeted SpinQuant rotations | Medium | Existing W16A8 rotations may be screened as transfer evidence. The accepted endpoint needs an FP8-objective validation and may require matched FP8-targeted rotation learning. |
| Custom NVIDIA W4A8 backend | High | The repository's owned W4A8 CUDA code is correctness-oriented and previously slow. Turning it into full-model serving requires an optimized kernel and vLLM integration, which is a separate backend project. |
| Official alternative backend | Medium, but different scope | The official [ExecuTorch Llama example](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md) provides a reusable SpinQuant-style W4A8 route for supported Arm/mobile Llama models. It can be evaluated separately, but it does not validate Llama-2-13B serving on GH200. |

Therefore the fake-quant route remains a contained paper-aligned study, while
production-speed INT8 W4A8 on GH200 is not the selected follow-up. W4AFP8 is
the planned hardware-aligned route, but its quality and speed remain unproven
until the deployed-checkpoint gates pass.

Assessment sources: the locally supplied `Spin.pdf` paper (deployment
discussion and Table 1), the stable vLLM
[W4A8 recipe](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int8_w4a8/),
and the stable vLLM
[hardware compatibility table](https://docs.vllm.ai/en/stable/features/quantization/).
The recipe demonstrates export and load; the hardware table is the controlling
claim boundary for NVIDIA INT4 x INT8 acceleration.

## Next stages

1. Finish and accept the already submitted paper-aligned `SpinQuant_no_had`
   fake-quant chain (`5850956` -> `5850958`): apply post-learning GPTQ W4 to
   the accepted W16A8-trained rotations and evaluate W4A8KV16 on the matched
   held-out tokens.
2. Freeze that result as INT8 W4A8 floating-QDQ evidence. It is not evidence for
   FP8 activation quality or real deployment performance.
3. After, and only after, the current chain is accepted, start the joint QuaRot
   and SpinQuant W4AFP8 deployment plan in
   [`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md). Both methods must
   pass deployed-checkpoint PPL and matched full-model serving benchmarks.
4. Keep `SpinQuant_had` with online R3/R4 as a separate paper extension. It is
   not a prerequisite for the offline-only W4AFP8 deployment route.

The rotation-learning part of stage 1 has passed both full-model gates. The
W16A8 smoke/formal configs make the rotation objective explicit as
`activation_qdq`, require unquantized 16-bit weights, and use unclipped
asymmetric per-token A8 floating QDQ. A local tiny-Llama mechanism test
confirmed non-zero R1/R2 gradients and the expected runtime forms. Isambard
job `5848060` completed the one-step Llama-2-13B smoke in 3 minutes 5 seconds
with exit code `0:0`. It produced a non-zero maximum rotation gradient of
`0.120179`; the accepted smoke rotation SHA256 is
`61b7d1f2b8eee6ad15584624d3be8d44b19ad858453cda113a83e2db08a0f6d8`.

The formal content-gated entry point is
`scripts/run_isambard_spinquant_llama2_13b_w16a8_100step.sbatch`. Formal job
`5848547` completed in 2 hours 9 minutes 15 seconds with exit code `0:0`. Its
first/last ten-step mean losses were `1.651698` and `1.561623`; the best loss
was `1.32231` at step 94. Final R1/R2 orthogonality errors were `1.3709e-6`
and `5.96e-7`. The accepted rotation tensor SHA256 is
`62cbc73e26d8993f068331e924c19738a40bc89813e680316d1978df3d452fda`.
Completion remains training evidence only. The post-learning GPTQ W4A8
fake-quant smoke/formal jobs `5850956` and `5850958` have been submitted as a
same-revision dependency chain, but no result is accepted until their artifacts
and full held-out metrics are validated. Checkpoint export and backend
validation remain later, separate gates.

The SpinQuant fake-quant stages remain independent of the completed QuaRot
deployment checkout and artifacts. The later joint W4AFP8 deployment work can
reuse the accepted vLLM process and provenance gates without modifying the
completed QuaRot evidence, but it cannot reuse W4A16 or INT8 W4A8 fake-quant
results as W4AFP8 accuracy evidence.
