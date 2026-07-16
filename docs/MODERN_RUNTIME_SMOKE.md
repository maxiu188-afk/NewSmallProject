# Modern runtime smoke

## Purpose

This records a local API-compatibility check on a newer Python, PyTorch, and
Transformers stack. It verifies the portable Python and Transformers paths only;
it is not CUDA-extension, kernel, memory, throughput, or server evidence.

## Environment and command

- Date: 2026-07-16
- Platform: macOS arm64, Apple M5; CPU execution (`mps` was available but not used)
- Python: 3.11.15
- PyTorch: 2.13.0
- Transformers: 5.14.1
- Dependencies: `requirements-modern-smoke.txt`

```bash
.venv-modern-smoke/bin/python -m unittest discover -v
.venv-modern-smoke/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/synthetic_llama_w4a4kv4_smoke.json \
  --output results/pipeline-synthetic-w4a4kv4-modern/result.json
.venv-modern-smoke/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local_quarot_w4a4kv4.json \
  --output results/pipeline-smollm2-135m-quarot-w4a4kv4-modern/result.json
```

## Result

All 26 tests passed. The random-GQA F5 smoke reported mean/max absolute logit
errors of `0.1140513420` / `0.5767568946`; the offline pinned SmolLM2-135M F5
smoke reported `5.0666241646` / `32.3876991272`. These agree with the earlier
Python 3.9 / PyTorch 2.2.1 smoke to the shown precision.

Transformers 5 deprecates the `torch_dtype` keyword in favor of `dtype`. The
pipeline now selects `dtype` for Transformers 5 and retains `torch_dtype` for
Transformers 4.x. Tokenizer loading no longer receives a model dtype keyword.

## Boundary

This check does not establish Linux CUDA, `nvcc`, CUDA-extension, or custom
kernel compatibility. The server preflight and matched CUDA/PyTorch build gate
remain mandatory before a formal experiment.
