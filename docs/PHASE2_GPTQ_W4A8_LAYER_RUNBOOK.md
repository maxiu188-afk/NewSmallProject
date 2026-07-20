# Phase-2 formal-GPTQ W4A8 Llama layer smoke

This is the next correctness gate after the formal F4 GPTQ packed-linear
export.  It runs one selected `q_proj` through the owned CUDA W4A8 module
inside an actual rotated Llama-2-13B decoder layer.  It is deliberately not a
full packed checkpoint, PPL, KV4, or performance run.

## 0. Start the Pod and synchronize code (Mac)

Start a CUDA Pod in the same RunPod region and attach the **existing**
persistent volume.  Copy its displayed SSH host and port into the variables
below.  Do not create a second model cache or virtual environment on the
persistent volume.

Run this block on the Mac from the repository root.  It transfers the commits
that the persistent server checkout does not yet have; it is necessary because
the server checkout is intentionally not configured to authenticate to GitHub.

```bash
cd /Users/xiuma/Desktop/work/NewSmallProject

POD_HOST='replace-with-RunPod-host'
POD_PORT='replace-with-RunPod-port'
POD_KEY="$HOME/.ssh/id_ed25519_Windows"
SERVER_BASE='793337e034d59f17720050602f3b6333d94fa1da'
BUNDLE_PATH='/tmp/newsmallproject-phase2-layer.bundle'

ssh-add --apple-use-keychain "$POD_KEY"
git status --short
git bundle create "$BUNDLE_PATH" HEAD "^${SERVER_BASE}"
git bundle list-heads "$BUNDLE_PATH"

ssh -p "$POD_PORT" -i "$POD_KEY" root@"$POD_HOST" 'mkdir -p /workspace/transfer'
scp -P "$POD_PORT" -i "$POD_KEY" "$BUNDLE_PATH" \
  root@"$POD_HOST":/workspace/transfer/newsmallproject-phase2-layer.bundle
ssh -p "$POD_PORT" -i "$POD_KEY" root@"$POD_HOST" '
  cd /workspace/NewSmallProject &&
  test -z "$(git status --porcelain)" &&
  git fetch /workspace/transfer/newsmallproject-phase2-layer.bundle \
    HEAD:refs/remotes/bundle/phase2-layer &&
  git merge --ff-only refs/remotes/bundle/phase2-layer &&
  git rev-parse HEAD
'
```

The final revision must be the current local `HEAD`.  Stop here if the server
working tree is not clean or the merge cannot fast-forward; do not use a merge
commit or overwrite the persistent checkout.

## Persistent inputs

The Pod must mount the existing project volume at `/workspace/NewSmallProject`.
Before starting work, verify these checked artifacts on that volume:

| File | Required SHA-256 |
|---|---|
| `results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.pt` | `573feb9812f7e002b53a22fbc993ba79e9f7822f76833630db882d548585c9a6` |
| `results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.json` | `4588fa52bdabb836812c062a7e8a54ef1576759ac8893f362491664c1a86f0bc` |

The cache root is `/workspace/NewSmallProject/.cache/huggingface`; do not
download a second model copy.  The model revision remains
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1`.

Run this next block in the Pod's shell to confirm the persistent inputs before
creating any container-disk environment:

```bash
cd /workspace/NewSmallProject
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
test -d .cache/huggingface/models--meta-llama--Llama-2-13b-hf
sha256sum results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.pt
sha256sum results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.json
```

The two hashes must match the table above.  Do not continue if either artifact
is missing or differs.

## Container-disk environment

Create the environment on the container disk, not `/workspace`.  The base
image must already provide the selected CUDA PyTorch build.  On the recorded
RTX 6000 Ada image this was PyTorch `2.8.0+cu128` with CUDA toolkit 12.8.

```bash
cd /workspace/NewSmallProject
VENV_PATH='/opt/newsmallproject/venvs/w4a8-cu128'
test -x "$VENV_PATH/bin/python" || python3 -m venv --system-site-packages "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install -r requirements-runpod-eval.txt

export HF_HOME=/workspace/NewSmallProject/.cache/huggingface
export HF_HUB_CACHE=/workspace/NewSmallProject/.cache/huggingface
"$VENV_PATH/bin/python" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY

"$VENV_PATH/bin/python" scripts/runpod_preflight.py \
  --output results/phase2-w4a8-layer/preflight.json
"$VENV_PATH/bin/python" scripts/run_w4a8_cuda_smoke.py \
  --output results/phase2-w4a8-layer/w4a8-kernel-smoke.json
```

`repro.w4a8_cuda` prepends the invoked virtual environment's `bin` directory
to `PATH` before compiling the extension, so detached `tmux` sessions can find
the pinned `ninja` executable without activating the environment shell.

## Fixed-token layer gate

Run only after the preflight and CUDA kernel smoke pass.  The script loads one
model, applies the recorded QuaRot rotation, and performs two sequential
forwards over the same `[1, 16]` deterministic token IDs:

1. `PackedW4A8ReferenceLinear` uses the exported signed W4, FP32 scales,
   act-order input permutation, and per-token A8 rule through an independent
   floating oracle.
2. `W4A8Linear` replaces that oracle with the owned CUDA int32-accumulator
   kernel using identical packed tensors.

```bash
/opt/newsmallproject/venvs/w4a8-cu128/bin/python \
  scripts/run_llama_gptq_w4a8_layer_smoke.py \
  --config configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_gptq.json \
  --artifact results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.pt \
  --artifact-report results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.json \
  --output results/phase2-w4a8-layer/llama2-13b-f4-qproj-layer-smoke.json \
  --sequence-length 16 \
  | tee results/phase2-w4a8-layer/llama2-13b-f4-qproj-layer-smoke.log

/opt/newsmallproject/venvs/w4a8-cu128/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("results/phase2-w4a8-layer/llama2-13b-f4-qproj-layer-smoke.json")
result = json.loads(path.read_text(encoding="utf-8"))
assert result["layer_output"]["allclose"], result["layer_output"]
assert result["logits"]["allclose"], result["logits"]
assert result["replacement"]["input_permutation_applied_before_a8"]
print("PHASE-2 LAYER GATE PASSED", result["layer_output"], result["logits"])
PY
```

The result must report finite layer/logit tensors and `allclose: true` for
both, with the predeclared `atol=2e-4` and `rtol=1e-5`.  It must also state:

- selected `q_proj`: packed signed W4 with FP32 group scales and per-token A8;
- all other linears: BF16;
- K/V: BF16;
- act-order input permutation applied before A8 quantization.

## Recovery, verification, and stop (Mac)

Run this on the Mac after the JSON assertion passed.  Reuse the `POD_HOST`,
`POD_PORT`, and `POD_KEY` values from step 0, or set them again before running
the block.

```bash
cd /Users/xiuma/Desktop/work/NewSmallProject
mkdir -p results/phase2-w4a8-layer

scp -P "$POD_PORT" -i "$POD_KEY" \
  root@"$POD_HOST":/workspace/NewSmallProject/results/phase2-w4a8-layer/preflight.json \
  results/phase2-w4a8-layer/
scp -P "$POD_PORT" -i "$POD_KEY" \
  root@"$POD_HOST":/workspace/NewSmallProject/results/phase2-w4a8-layer/w4a8-kernel-smoke.json \
  results/phase2-w4a8-layer/
scp -P "$POD_PORT" -i "$POD_KEY" \
  root@"$POD_HOST":/workspace/NewSmallProject/results/phase2-w4a8-layer/llama2-13b-f4-qproj-layer-smoke.json \
  results/phase2-w4a8-layer/
scp -P "$POD_PORT" -i "$POD_KEY" \
  root@"$POD_HOST":/workspace/NewSmallProject/results/phase2-w4a8-layer/llama2-13b-f4-qproj-layer-smoke.log \
  results/phase2-w4a8-layer/

shasum -a 256 results/phase2-w4a8-layer/*.json
```

The JSON files are ignored by Git; retain their hashes and raw log for the
result record.  Stop the Pod from the RunPod dashboard after successful local
recovery.  Do not use `shutdown` inside the container: Pod lifecycle is not
managed by systemd there.

If the gate fails, recover the same files and stop.  Do not run PPL, KV4, or
timing experiments.  A pass only establishes fixed-token selected-linear
integration; the next task is a self-describing multi-linear/full-model packed
checkpoint plan.
