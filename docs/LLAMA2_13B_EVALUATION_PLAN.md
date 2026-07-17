# Llama-2-13B formal evaluation plan

## Scope and order

The primary model is `meta-llama/Llama-2-13b-hf`, not Qwen. The first server
result is the full-precision WikiText-2 baseline. Only after it completes with
pinned model and dataset revisions will fake quantization be aligned with the
QuaRot paper: first an RTN control, then GPTQ with the same calibration corpus,
sample count, sequence length, seed, and bit widths across compared rows.

The portable pipeline's current W-bit operation is RTN-style QDQ. It must not
be labelled GPTQ. The GPTQ stage uses the upstream fake-quant behaviour as a
reference and records every calibration and evaluation choice separately.

## Server preparation and authentication

On the RunPod persistent volume, keep the Hugging Face cache and generated
records outside the Git worktree:

```bash
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE=/workspace/hf-cache
hf auth login
```

Authenticate with the Hugging Face account whose access request for Llama-2
has been accepted. Do not put the token in a shell history, tracked config,
result JSON, or Git commit.

Install the selected CUDA PyTorch wheel first, then the Python-level runtime:

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
python3 -m pip install -r requirements-runpod-eval.txt
```

## Pinned full-precision baseline

Use `configs/pipeline/llama2_13b_wikitext2_bf16_baseline.json`. It is an
unresolved template and intentionally cannot be run directly. Resolve the
model and dataset commits first, then run only the resulting local-file config:

```bash
python3 scripts/prepare_hf_model.py \
  configs/pipeline/llama2_13b_wikitext2_bf16_baseline.json \
  --output results/model-prep/llama2-13b-manifest.json \
  --write-resolved-config results/model-prep/llama2-13b-model-pinned.json

python3 scripts/prepare_hf_dataset.py \
  results/model-prep/llama2-13b-model-pinned.json \
  --output results/data-prep/wikitext2-manifest.json \
  --write-resolved-config results/configs/llama2-13b-wikitext2-bf16-pinned.json

python3 scripts/run_quarot_pipeline.py \
  results/configs/llama2-13b-wikitext2-bf16-pinned.json \
  --output results/llama2-13b-wikitext2-bf16/result.json
```

The pipeline runs this F0 baseline without cloning the model. This is necessary
on the A40: one BF16 13B model is feasible, whereas reference and candidate
copies together are not a safe 48 GB workload.

## Gate before GPTQ

The baseline record must contain the model and dataset commits, command,
environment snapshot, token count, mean NLL, and PPL. Review it before
downloading calibration data or starting GPTQ. GPTQ results must compare F0,
F1/F2, F3/F4, and F5 on the same pinned WikiText-2 protocol; no synthetic
token result substitutes for this gate.
