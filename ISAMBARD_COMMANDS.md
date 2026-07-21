# Isambard Command Reference

This is the concise operational reference for the Llama-2-13B GPTQ/W4A8
deployment work. It is intentionally separate from the detailed experiment
runbooks in `docs/`: use this file for login, setup, cache, and Slurm commands.

## On the local machine

Refresh the Clifton SSH certificate (it normally expires after about 12 hours),
then connect to the `u6rt` project:

```bash
clifton auth
clifton ssh-config write
ssh u6rt.aip2.isambard
```

Do not use the older `b6u` project for this work.

## Project paths and current environment

After logging in, use the project copy and persistent paths below. Do not use
node-local storage for the virtual environment, model cache, or results.

```bash
cd "$HOME/NewSmallProject"
export VENV_PATH="$HOME/.venvs/newsmallproject-w4a8"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
```

As of 2026-07-20, the environment and assets are ready directly in the login
environment. The attempted batch setup job `5729880` failed only because the
ARM64 CUDA index does not publish `torch==2.8.0`; its dependent prefetch job
`5729928` was cancelled. Do not resubmit either job.

The installed runtime is `torch==2.9.0+cu128`; this is intentionally different
from the earlier x86_64 RunPod runtime (`2.8.0+cu128`).

## Hugging Face authentication

Authentication is already configured for this account. Check it without
printing or copying any token:

```bash
"$HOME/.venvs/hf-auth/bin/hf" auth whoami
```

After the full environment exists, the equivalent command is:

```bash
"$VENV_PATH/bin/hf" auth whoami
```

To authenticate again after token expiry, run the interactive command and type
the token only into the Isambard terminal:

```bash
"$HOME/.venvs/hf-auth/bin/hf" auth login
```

## Environment and cache checks

The environment, pinned model revision, and Wikitext-2 train/test splits are
already cached. Run this on the login node to confirm the persistent files:

```bash
cd "$HOME/NewSmallProject"
module load cray-python/3.11.7 cuda/12.6

"$VENV_PATH/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda)'

test -d "$HF_HUB_CACHE/models--meta-llama--Llama-2-13b-hf"
du -sh "$VENV_PATH" "$HF_HOME"
```

The login node has no GPU allocation; use the interactive diagnostic below for
`torch.cuda.is_available()` and device checks.

The required model revision is
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1`. The cache is rebuildable local
state: do not add it to Git or copy it back from RunPod.

## Direct rebuild commands

Only use these commands for an intentional clean rebuild. Configuration is
performed directly on the login node; do not submit it as a Slurm job.

```bash
cd "$HOME/NewSmallProject"
module load cray-python/3.11.7 cuda/12.6
VENV_PATH="$HOME/.venvs/newsmallproject-w4a8"
test -x "$VENV_PATH/bin/python" || python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0"
"$VENV_PATH/bin/python" -m pip install -r requirements-runpod-eval.txt
```

Prefetch the pinned model serially, after `hf auth whoami` succeeds. Do not use
the default eight download workers on the login node:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_DISABLE_XET=1
"$VENV_PATH/bin/hf" download meta-llama/Llama-2-13b-hf \
  --revision 5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1 \
  --cache-dir "$HF_HUB_CACHE" --max-workers 1
```

## Monitor and inspect jobs

Replace `<JOBID>` with a Slurm job ID.

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.12M %.12l %R"
sacct --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80

tail -f "w4a8-env-<JOBID>.out"
cat "w4a8-env-<JOBID>.err"
```

`PENDING (Priority)` is a normal queue state. `PENDING (Dependency)` means a
job is waiting for its prerequisite to succeed. Check `sacct` before changing
an existing experiment job.

## Submit the Phase-2 W4A8 layer gate

This is the next formal GPU task. It runs an Isambard preflight, the owned CUDA
int32 kernel smoke, and the fixed-token Llama q_proj layer/logits comparison.
It writes an input/source manifest plus three JSON reports under
`results/phase2-w4a8-layer/`.

```bash
cd "$HOME/NewSmallProject"
sbatch scripts/run_isambard_w4a8_layer_gate.sbatch
```

Monitor the returned job ID and inspect the batch log:

```bash
squeue -j <JOBID> -o "%.18i %.9T %.10M %.30R"
tail -f "w4a8-layer-gate-<JOBID>.out"
cat "w4a8-layer-gate-<JOBID>.err"
```

Success requires `ISAMBARD_PHASE2_LAYER_GATE_PASSED` in the output and
`passed: true` in `isambard-llama2-13b-f4-qproj-layer-smoke.json`. The job does
not establish a packed full model, PPL, KV4, or performance result.

### Batch-script path rule and current submission

Slurm executes a spool copy of an `.sbatch` file. A batch script must therefore
use `SLURM_SUBMIT_DIR` to locate the checkout; do not derive the project root
solely from `BASH_SOURCE`. The checked-in layer-gate script follows this rule
and prints these stage markers in its output:

```text
W4A8_LAYER_GATE_STAGE=module_load
W4A8_LAYER_GATE_STAGE=inputs
W4A8_LAYER_GATE_STAGE=source_manifest
W4A8_LAYER_GATE_STAGE=preflight
W4A8_LAYER_GATE_STAGE=kernel_smoke
W4A8_LAYER_GATE_STAGE=layer_smoke
W4A8_LAYER_GATE_STAGE=assert_result
```

The pre-repair jobs `5732906` and `5735242` stopped at `inputs` because they
looked for the packed artifact under Slurm's spool directory. The repaired job
`5739260` was submitted on 2026-07-21; it was pending at submission time. A
pending job is not a failed GPU run and is not evidence of numerical success.

## Prepared full-decoder gate (do not submit during the current service issue)

The next implementation is checked in as
`scripts/run_isambard_w4a8_full_model_gate.sbatch`. It streams all 280 GPTQ
decoder linears to a sharded packed checkpoint, validates every shard, compares
all decoder-layer outputs and logits between the packed oracle and CUDA path,
and runs a short BF16-K/V generation smoke. It does not run PPL, KV4, timing,
throughput, or memory measurements.

Keep the existing selected-linear job and review it when Isambard is healthy.
Before submitting the larger gate, synchronize the reviewed repository revision
and ask Slurm to validate its request:

```bash
cd "$HOME/NewSmallProject"
sbatch --test-only scripts/run_isambard_w4a8_full_model_gate.sbatch
sbatch scripts/run_isambard_w4a8_full_model_gate.sbatch
```

The `--test-only` step must accept the requested 24-hour wall time; it has not
been validated while the service is unhealthy. Success requires
`ISAMBARD_PHASE2_FULL_MODEL_GATE_PASSED`. Detailed artifact and failure rules
are in `docs/PHASE2_GPTQ_W4A8_FULL_MODEL_RUNBOOK.md`.

## Short interactive GPU diagnostic

Use this only for a quick interactive diagnostic; it ends when the shell exits.
Formal W4A8 experiments should use a dedicated checked-in batch script and
retain JSON reports plus Slurm logs.

```bash
srun --account=brics.u6rt --partition=workq --nodes=1 --gpus=1 \
  --time=00:10:00 --pty /bin/bash --login

module load cray-python/3.11.7 cuda/12.6
cd "$HOME/NewSmallProject"
export VENV_PATH="$HOME/.venvs/newsmallproject-w4a8"
"$VENV_PATH/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))'
```

## Scope reminder

The recovered Phase-2 result validates a selected `q_proj` W4A8 replacement at
decoder-layer and logits level. A complete packed model, PPL, KV4, and
performance evidence remain separate future gates; do not claim them from the
environment or cache setup alone.
