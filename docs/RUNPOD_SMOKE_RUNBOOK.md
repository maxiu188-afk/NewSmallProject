# RunPod smoke runbook

> Historical status: CUDA smoke completed (2026-07-16). Preflight, a 16-bit
> rotation check, and a matched synthetic-input W4A4KV4 smoke completed on an
> A40. Those results do not authorize a benchmark or a text-accuracy claim.
> A separate RTX 6000 Ada CUDA 12.4 GPTQ session is now active; its pending
> artifacts and review boundary are recorded in `RUNPOD_GPTQ_SESSION.md`.

The selected launch resources are recorded in `RUNPOD_LAUNCH_SPEC.md`.

This runbook starts only after the local reference checks pass. It does not
authorize a benchmark before compatibility and numerical correctness gates.

## Recorded progress

- Gate A preflight passed on an A40 (`sm_86`) with CUDA compiler `12.8.93` and
  PyTorch `2.11.0+cu128`.
- The portable CUDA path passed a 16-bit SmolLM2-135M rotation check and a
  matched synthetic-input W4A4KV4 naive/QuaRot smoke.
- The unmodified upstream editable build was attempted. CMake configuration
  completed, but the legacy nested editable install failed under modern pip
  build isolation because its setup script imports PyTorch. The failure was
  logged; no upstream source patch was applied and no native kernel result is
  claimed.

See `RUNPOD_FAKE_QUANT_RESULTS.md` for exact versions and metrics.

## A. Preflight: no model required

1. Place this project and the unchanged `QuaRot/` reference checkout on the
   RunPod volume. Do not copy a virtual environment from macOS.
2. From the project root, create a preflight record before installing or
   patching anything:

   ```bash
   python3 scripts/runpod_preflight.py --output results/runpod-preflight/preflight.json
   ```

3. Inspect the JSON file. It must identify GPU model/count, compute capability,
   driver, CUDA compiler, Python, current package versions, and the upstream
   commit. Keep it unchanged as evidence.
4. Apply [the CUDA/PyTorch compatibility policy](CUDA_PYTORCH_COMPATIBILITY_POLICY.md):
   record both `nvcc --version` and `torch.version.cuda`, then select a matched
   `cu128` or `cu130` environment. The CUDA value shown by `nvidia-smi` alone
   is insufficient for a custom extension build.
5. Compare compute capability with the upstream `setup.py` target list
   (`sm_75`, `sm_80`, `sm_86`). An unsupported architecture is a recorded
   compatibility finding. Do not edit build flags yet.

**Gate A passes when:** the preflight JSON exists and the source revision,
hardware, and initial package state are known.

## B. Pinned environment and reference build

1. Create a fresh Linux virtual environment on RunPod.
2. Initialize the exact upstream submodules and record their SHAs.
3. Install the pinned reference requirements only in that environment.
4. Attempt the unmodified upstream build once and save complete stdout/stderr.

**Gate B passes when:** either the unmodified build succeeds, or the exact
failure is captured. Only then may a minimal patch be proposed under `patches/`.

## C. Minimal algorithm smoke

Before PPL or GPTQ:

1. Run the no-quantization rotation-equivalence check on a fixed tiny/random
   LLaMA-shaped configuration.
2. Run a fixed-input numerical comparison of QDQ reference values against the
   framework path.
3. Record tolerances and maximum errors in a dedicated result directory.

**Gate C passes when:** rotation with all precisions at 16-bit preserves logits
within declared tolerance. A failure blocks F0–F5 accuracy experiments.

## D. Accuracy smoke, then full matrix

The selected evaluation dataset is `wikitext2`, matching the upstream
fake-quant defaults. The next primary model is
`meta-llama/Llama-2-13b-hf`. Authenticate the RunPod environment with the
approved Hugging Face account before downloading it. Resolve and record the
exact model, tokenizer, and dataset revisions, then write a per-run config.

`configs/templates/llama2_fake_quant_matrix.json` remains a historical
LLaMA-2 reference template. Do not materialize it directly for Qwen. After
the Qwen equivalence gate, create a Qwen-specific matrix with the same F0–F5
control logic and resolved model/dataset revisions, then review the independent
configs before execution.

Run BF16/FP16 and fake-quant experiments with identical model, data, seed,
calibration count, and sequence length. Only after the smoke run completes may
the F0–F5 matrix in the project plan be scaled up.

## E. Kernel path

Run packed-W4 linear and int4 KV numerical checks before any timing. Benchmark
prefill and decode separately, with warm-up, CUDA synchronization, repeated
samples, raw timings, and peak allocated memory. Label results by actual
execution mode, never merely by checkpoint storage format.
