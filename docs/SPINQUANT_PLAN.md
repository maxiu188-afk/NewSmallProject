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

This formal configuration and runner have not been executed on CUDA. A
same-revision one-step smoke must pass before the 100-step job is submitted.
The paired smoke configurations retain sequence length 2048 and all model,
quantization, runtime, and optimizer settings; they reduce only the calibration
set from 800 to 8 sequences and the optimizer updates from 100 to 1.

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

The formal entry point is
`scripts/run_isambard_spinquant_llama2_13b_100step.sbatch`. It requests six
hours based on the observed one-step runtime and validates the completed smoke
result plus every smoke controlling-source hash before training. Success
requires all 100 loss and non-zero gradient records, bounded orthogonality,
the expected R1/R2 shapes, and a checksummed SafeTensors artifact.

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
```

## Next stages

1. Materialize and validate the pinned WikiText-2 artifact in the GPU
   environment, then run a same-revision one-step 13B training smoke before
   the 100-step job.
2. Reproduce the Llama-2-7B fake-quant matrix for `SpinQuant_no_had` and
   `SpinQuant_had`, followed by the matched Llama-2-13B comparison.
3. Keep paper-protocol results separate from the same-token QuaRot comparison.
4. For real deployment, retain only offline-fusible learned R1/R2, apply
   group-128 GPTQ W4A16, and reuse the standard compressed-tensors/vLLM route.

The SpinQuant fake-quant stages do not depend on the queued deployed-checkpoint
PPL job. A failure in that job would block only the later reuse of the vLLM
quality gate, not R1/R2 development or fake-quant experiments.
