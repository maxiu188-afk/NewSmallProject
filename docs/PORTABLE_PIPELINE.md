# Portable QuaRot pipeline

`scripts/run_quarot_pipeline.py` is configuration-driven and avoids embedding a
particular model or dataset in Python code. It runs on Windows and macOS through
the same `python` entry point and uses `pathlib` for local data paths.

## Configuration choices

- `model.kind: random_config` creates an architecture from local configuration
  only; it is used for code-correctness smoke tests.
- `model.kind: pretrained` loads the model ID and optional revision supplied in
  JSON. `local_files_only: true` prevents an accidental model download.
- `data.source` can be `synthetic`, `jsonl_text`, or `huggingface_text`. Only
  the latter needs the optional `datasets` package.
- `runtime.device` is explicit. If an unavailable device is requested and
  `allow_fallback` is false, the run fails rather than silently switching CPU,
  MPS, or CUDA.

## QuaRot adapter status

The adapter is generic over LLaMA model dimensions and supports grouped-query
attention in its residual and V/O transformations. It derives all dimensions
from `model.config`; no model ID is special-cased.

Implemented: RMSNorm fusion, residual rotation (`hadamard` or seeded `random`),
V/O compensation, optional power-of-two MLP online Hadamard, W-bit QDQ, and
A-bit per-token QDQ.

When a LLaMA-family checkpoint ties input embeddings and the output head, the
adapter copies the output head before reparameterization. This is required for
exactness: the input embedding and output projection undergo different valid
transformations after final-RMSNorm fusion.

Explicitly not yet implemented: Q/K post-RoPE rotation. The portable adapter
now supports both power-of-two online MLP Hadamards and the `12 x power-of-two`
structure used by SmolLM2-135M (`1536 = 12 x 128`). It is still a partial
QuaRot pipeline because Q/K rotation and KV-cache handling are absent.

## Preparing a pretrained model safely

Use the separate preparation command once for each model revision. It downloads
the repository selected by the configuration, records the resolved immutable
commit in a local manifest, and writes a pinned copy of the configuration.
The runner itself never downloads when its configuration uses
`local_files_only: true`.

Both commands set the Hub cache explicitly to `<project>/.cache/huggingface`,
so a snapshot prepared on one operating system is discovered by the offline
runner on that same machine. The cache itself remains machine-local and is not
portable between operating systems.

macOS/Linux:

```bash
.venv-smollm/bin/python scripts/prepare_hf_model.py \
  configs/pipeline/smollm2_135m_local.json \
  --output results/model-prep/smollm2-135m-manifest.json \
  --write-resolved-config results/model-prep/smollm2-135m-pinned.json
```

Windows PowerShell:

```powershell
.venv-smollm\Scripts\python.exe scripts\prepare_hf_model.py `
  configs\pipeline\smollm2_135m_local.json `
  --output results\model-prep\smollm2-135m-manifest.json `
  --write-resolved-config results\model-prep\smollm2-135m-pinned.json
```

Review the generated manifest and use the pinned configuration for checks. The
manifest and generated configuration are ignored as local artifacts; after
review, copy only the 40-character commit into a tracked experiment config if
that exact revision becomes part of the recorded protocol.

## Local commands

macOS/Linux:

```bash
.venv-smoke/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/synthetic_llama_smoke.json \
  --output results/pipeline-synthetic-smoke/result.json

# Same architecture with W4A4 fake quantization enabled
.venv-smoke/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/synthetic_llama_w4a4_smoke.json \
  --output results/pipeline-synthetic-w4a4-smoke/result.json
```

Windows PowerShell after creating an equivalent virtual environment:

```powershell
.venv-smoke\Scripts\python.exe scripts\run_quarot_pipeline.py `
  configs\pipeline\synthetic_llama_smoke.json `
  --output results\pipeline-synthetic-smoke\result.json
```

Validate the selected model configuration without downloading it:

```bash
.venv-smoke/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local.json --validate-only
```
