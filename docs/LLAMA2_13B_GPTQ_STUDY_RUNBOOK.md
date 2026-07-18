# Llama-2-13B GPTQ component and calibration study

## Objective

This is the next server-side accuracy study after the reviewed QuaRot F4 GPTQ
record. It has two controlled questions:

1. Which QuaRot transformations account for the F4 GPTQ result?
2. Does the required GPTQ calibration size differ between unrotated F3 and
   full QuaRot F4?

It deliberately excludes K/V 4-bit QDQ (F5), packed weights, custom kernels,
latency, throughput, and memory measurements. All rows remain floating-point
fake quantization on the pinned Llama-2-13B and WikiText-2 protocol.

## Matrix

All rows use symmetric W4 GPTQ (group/block size 128, 1% damping and
activation ordering), A4 evaluation, K/V at 16-bit, seed 0, and 2048-token
sequences. Calibration always reads the pinned WikiText-2 `train` split;
evaluation always reads the pinned `test` split.

| Order | Result ID | Calibration sequences | Transformation under test |
|---:|---|---:|---|
| 1 | `naive-f3-cal128` | 128 | Unrotated F3 control |
| 2 | `quarot-residual-cal128` | 128 | Residual Hadamard only |
| 3 | `quarot-residual-vo-cal128` | 128 | Add V/O rotation |
| 4 | `quarot-residual-vo-mlp-cal128` | 128 | Add online MLP Hadamard |
| 5 | `quarot-f4-cal128` | 128 | Add post-RoPE Q/K rotation; complete F4 |
| 6 | `naive-f3-cal32` | 32 | F3 calibration-size row |
| 7 | `quarot-f4-cal32` | 32 | F4 calibration-size row |
| 8 | `naive-f3-cal64` | 64 | F3 calibration-size row |
| 9 | `quarot-f4-cal64` | 64 | F4 calibration-size row |

Rows 1 and 5 are also the 128-sequence endpoints of the calibration study,
so the complete design has nine runs, not eleven.

## Preflight and launch

Use the same RTX 6000 Ada CUDA 12.4 environment and persistent project/cache
layout as the reviewed F4 GPTQ run. Do not run the study until the tiny
end-to-end CUDA GPTQ smoke has passed again in that exact runtime.

```bash
cd /workspace/NewSmallProject
CUDA_PYTHON=/opt/quarot-venv-cu124-study/bin/python

$CUDA_PYTHON scripts/run_tiny_llama_gptq_smoke.py \
  --output results/llama2-13b-wikitext2-gptq-study/tiny-gptq-smoke.json
$CUDA_PYTHON scripts/run_llama2_13b_gptq_study.py --validate-only
$CUDA_PYTHON scripts/run_llama2_13b_gptq_study.py --dry-run
tmux new-session -d -s gptq-study \
  "$CUDA_PYTHON scripts/run_llama2_13b_gptq_study.py --resume"
```

The launcher stops at the first failed row. It writes one result JSON and one
append-only log per row under
`results/llama2-13b-wikitext2-gptq-study/`. Re-run the same command with
`--resume` only after inspecting a failure: it skips an existing row solely
when its JSON has finite metrics, uses GPTQ, and reports the configured number
of calibration batches.

## Review and interpretation

Before accepting any row, verify its exit code, log, finite metrics, 280 GPTQ
linear layers, model/dataset revisions, evaluation token count, calibration
split, and calibration sequence/token count. Recover the raw JSON/log files
from `/workspace` and checksum them before shutting down the server.

The component table compares rows 1--5 only. The calibration table compares
F3 and F4 at like-for-like 32, 64, and 128 sequence counts. Do not compare a
calibration-size row to a component row as though calibration were controlled.
