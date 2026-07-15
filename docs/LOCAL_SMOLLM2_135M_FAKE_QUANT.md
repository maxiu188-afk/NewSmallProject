# SmolLM2-135M local fake-quant matrix

## What this verifies

All five cases use the same pinned SmolLM2-135M snapshot, one fixed synthetic
CPU batch (1 x 32 tokens), float32 loading, and the same QDQ implementation.
Weight QDQ is symmetric per output channel; activation QDQ is symmetric
per-token immediately before every `nn.Linear`. No calibration data, KV-cache
QDQ, GPTQ/AWQ, packing, or integer GEMM is involved.

`QuaRot` below means the current portable subset: seeded random residual
rotation, V/O compensation, and the 12 x 128 online MLP transform. Q/K
post-RoPE rotation remains disabled.

## Commands

Each configuration can be invoked through the same portable entry point. For
example, on macOS/Linux:

```bash
.venv-smollm/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local_quarot_w4a4.json \
  --output results/pipeline-smollm2-135m-quarot-w4a4/result.json
```

Use the `naive` or `quarot` and `w4` or `w4a4` filename variants to run F1-F4.
On Windows, replace the interpreter path with
`.venv-smollm\Scripts\python.exe` and use PowerShell line continuation.

## Observed local smoke results

| Case | W/A bits | Mean absolute logit error | Max absolute logit error | Synthetic NLL |
| --- | --- | ---: | ---: | ---: |
| F0 FP32 reference | 16 / 16 | 0 | 0 | 8.830813 |
| F1 naive W4 | 4 / 16 | 5.812171 | 40.829163 | 10.479802 |
| F2 partial QuaRot W4 | 4 / 16 | 3.642915 | 32.597256 | 11.517881 |
| F3 naive W4A4 | 4 / 4 | 11.402394 | 56.492119 | 19.898615 |
| F4 partial QuaRot W4A4 | 4 / 4 | 5.086081 | 38.154541 | 13.054570 |

For this fixed local input, the partial QuaRot path reduces both reported logit
errors relative to naive QDQ. The NLL is computed on deterministic synthetic
IDs, not natural text, so it is only a code-path diagnostic and must not be
reported as perplexity, accuracy, or a reproduction of the QuaRot paper.

## Interpretation boundary

This is evidence that the pretrained-model rotation and W/A QDQ paths execute
in a controlled comparison. It is not evidence for full A4W4KV4, paper-level
accuracy, real low-bit arithmetic, memory reduction, or throughput. A later
RunPod experiment must use a pinned text dataset and report the full evaluation
contract separately.
