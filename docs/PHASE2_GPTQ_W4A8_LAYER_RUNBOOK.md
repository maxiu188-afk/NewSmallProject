# Phase-2 formal-GPTQ W4A8 Llama layer smoke

This is the next correctness gate after the formal F4 GPTQ packed-linear
export.  It runs one selected `q_proj` through the owned CUDA W4A8 module
inside an actual rotated Llama-2-13B decoder layer.  It is deliberately not a
full packed checkpoint, PPL, KV4, or performance run.

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

## Container-disk environment

Create the environment on the container disk, not `/workspace`.  The base
image must already provide the selected CUDA PyTorch build.  On the recorded
RTX 6000 Ada image this was PyTorch `2.8.0+cu128` with CUDA toolkit 12.8.

```bash
cd /workspace/NewSmallProject
python3 -m venv --system-site-packages /opt/newsmallproject/venvs/w4a8-cu128
/opt/newsmallproject/venvs/w4a8-cu128/bin/python -m pip install -r requirements-runpod-eval.txt
/opt/newsmallproject/venvs/w4a8-cu128/bin/python scripts/runpod_preflight.py \
  --output results/phase2-w4a8-layer/preflight.json
/opt/newsmallproject/venvs/w4a8-cu128/bin/python scripts/run_w4a8_cuda_smoke.py \
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
  --sequence-length 16
```

The result must report finite layer/logit tensors and `allclose: true` for
both, with the predeclared `atol=2e-4` and `rtol=1e-5`.  It must also state:

- selected `q_proj`: packed signed W4 with FP32 group scales and per-token A8;
- all other linears: BF16;
- K/V: BF16;
- act-order input permutation applied before A8 quantization.

## Recovery and boundary

Copy `preflight.json`, `w4a8-kernel-smoke.json`, and
`llama2-13b-f4-qproj-layer-smoke.json` back to the ignored local `results/`
tree and SHA-256 verify them before stopping the Pod.  If the gate fails, keep
the log and JSON, but do not run PPL, KV4, or timing experiments.  A pass only
establishes fixed-token selected-linear integration; the next task is a
self-describing multi-linear/full-model packed checkpoint plan.
