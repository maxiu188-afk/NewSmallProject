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
V/O compensation, Q/K per-head Hadamard immediately after RoPE, optional
power-of-two MLP online Hadamard, W-bit QDQ, A-bit per-token QDQ, K-cache QDQ,
and separate V-projection-output QDQ. K-cache QDQ supports either token-wise
groups across all heads (`k_group_size: -1`) or a group per head
(`k_group_size: head_dim`); the equivalent V option is `v_group_size`.

When a LLaMA-family checkpoint ties input embeddings and the output head, the
adapter copies the output head before reparameterization. This is required for
exactness: the input embedding and output projection undergo different valid
transformations after final-RMSNorm fusion.

The portable adapter supports both power-of-two online MLP Hadamards and the
`12 x power-of-two` structure used by SmolLM2-135M (`1536 = 12 x 128`). The
post-RoPE wrapper targets the ordinary Hugging Face LLaMA attention helper and
must be numerical-smoke-tested for every pinned Transformers version before a
pretrained result is accepted. It is fake quantization only: it neither packs
the cache nor replaces floating-point attention with an integer kernel.

The current newer-runtime API smoke (Python 3.11, PyTorch 2.13, Transformers 5)
is recorded in `archive/smokes/MODERN_RUNTIME_SMOKE.md`; the server still
requires its separate
Linux CUDA-extension compatibility gate.

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

# F5 algorithm smoke: Q/K post-RoPE rotation plus W4A4KV4 QDQ through
# sequential cached decoding (still floating-point QDQ, not a deployment run)
.venv-smoke/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/synthetic_llama_w4a4kv4_smoke.json \
  --output results/pipeline-synthetic-w4a4kv4-smoke/result.json

# Matched offline W4A4KV4 control and F5 path on the pinned SmolLM2-135M checkpoint.
.venv-smollm/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local_naive_w4a4kv4.json \
  --output results/pipeline-smollm2-135m-naive-w4a4kv4/result.json

.venv-smollm/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local_quarot_w4a4kv4.json \
  --output results/pipeline-smollm2-135m-quarot-w4a4kv4/result.json
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
