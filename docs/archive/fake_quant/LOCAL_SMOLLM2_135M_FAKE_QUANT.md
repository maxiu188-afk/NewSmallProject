# SmolLM2-135M local fake-quant matrix

## What this verifies

All seven cases use the same pinned SmolLM2-135M snapshot, one fixed synthetic
CPU batch (1 x 32 tokens), float32 loading, and the same QDQ implementation.
Weight QDQ is symmetric per output channel; activation QDQ is symmetric
per-token immediately before every `nn.Linear`. F5 additionally applies
per-head Q/K post-RoPE Hadamard, K-cache QDQ, V-projection-output QDQ, and
sequential cached decoding. No calibration data, GPTQ/AWQ, packing, or integer
GEMM is involved.

`QuaRot` below means the current portable subset: seeded random residual
rotation, V/O compensation, and the 12 x 128 online MLP transform. F5 also
uses Q/K post-RoPE rotation and K/V QDQ.

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

The matched local W4A4KV4 control and F5 configurations are:

```bash
.venv-smollm/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local_naive_w4a4kv4.json \
  --output results/pipeline-smollm2-135m-naive-w4a4kv4/result.json

.venv-smollm/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local_quarot_w4a4kv4.json \
  --output results/pipeline-smollm2-135m-quarot-w4a4kv4/result.json
```

## Observed local smoke results

| Case | W/A bits | Mean absolute logit error | Max absolute logit error | Synthetic NLL |
| --- | --- | ---: | ---: | ---: |
| F0 FP32 reference | 16 / 16 | 0 | 0 | 8.830813 |
| F1 naive W4 | 4 / 16 | 5.812171 | 40.829163 | 10.479802 |
| F2 partial QuaRot W4 | 4 / 16 | 3.642915 | 32.597256 | 11.517881 |
| F3 naive W4A4 | 4 / 4 | 11.402394 | 56.492119 | 19.898615 |
| F4 partial QuaRot W4A4 | 4 / 4 | 5.086081 | 38.154541 | 13.054570 |
| C5 naive W4A4KV4 control | 4 / 4 (K/V 4) | 9.061684 | 50.128185 | 18.223836 |
| F5 QuaRot W4A4KV4 | 4 / 4 (K/V 4) | 5.066625 | 32.387703 | 12.883306 |

For this fixed local input, the partial QuaRot path reduces both reported logit
errors relative to naive QDQ. At W4A4KV4, F5 reduces the C5 mean and maximum
logit errors by about 44% and 35%, respectively. The C5/F5 pair isolates the
added rotation path at the same bit widths. The NLL is computed on deterministic
synthetic IDs, not natural text, so it is only a code-path diagnostic and must
not be reported as perplexity, accuracy, or a reproduction of the QuaRot paper.

## Interpretation boundary

This is evidence that the pretrained-model rotation and W/A/K/V fake-QDQ paths
execute in a controlled comparison. It is not evidence for paper-level
accuracy, real low-bit arithmetic, memory reduction, or throughput. A later
RunPod experiment must use a pinned text dataset and report the full evaluation
contract separately.
