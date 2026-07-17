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

## Completed F0 record

The BF16 baseline completed on 2026-07-17 with model revision
`5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` and WikiText-2 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. It scored mean NLL
`1.6111028514668504` and PPL `5.008331629066348` over 331,614 next-token
targets (162 sequences of length 2048). The full result and its evaluation
boundary are recorded in [LLAMA2_13B_BF16_BASELINE.md](LLAMA2_13B_BF16_BASELINE.md).

## Server preparation and authentication

The current preparation and runner scripts explicitly use the ignored project
cache `.cache/huggingface`, overriding a caller-supplied `HF_HOME`. Until that
implementation changes, authenticate into that cache and keep it on the
persistent volume:

```bash
export HF_HOME=/workspace/NewSmallProject/.cache/huggingface
export HF_HUB_CACHE=/workspace/NewSmallProject/.cache/huggingface
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

For the completed download, `hf-xet 1.5.2` failed with
`DataHashHexParseError`; the successful retry used `HF_HUB_DISABLE_XET=1`.
This controls the transport only and does not change the evaluated model.

## RTN W4A4 gate before GPTQ

The next server session runs only the matched W4A4 pair, using the already
cached, pinned Llama-2-13B and WikiText-2 snapshots. These are RTN-style QDQ
experiments: no calibration set is used and they must not be called GPTQ.

```bash
export HF_HUB_DISABLE_XET=1
cd /workspace/NewSmallProject
python3 scripts/run_quarot_pipeline.py \
  configs/pipeline/llama2_13b_wikitext2_naive_w4a4_rtn.json \
  --output results/llama2-13b-wikitext2-rtn-w4a4/naive-f3.json

python3 scripts/run_quarot_pipeline.py \
  configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_rtn.json \
  --output results/llama2-13b-wikitext2-rtn-w4a4/quarot-f4.json

python3 scripts/compare_rtn_w4a4.py \
  --naive results/llama2-13b-wikitext2-rtn-w4a4/naive-f3.json \
  --quarot results/llama2-13b-wikitext2-rtn-w4a4/quarot-f4.json \
  --output results/llama2-13b-wikitext2-rtn-w4a4/comparison.json
```

The comparison command only passes when both result files use the same pinned
model/data/evaluation protocol and seed, are respectively unrotated F3 and
Hadamard-rotated F4, keep KV quantization disabled, and contain finite
metrics. It prints the PPL/NLL difference; it deliberately does not impose an
invented PPL threshold. Inspect that difference before deciding that QuaRot is
accurate enough to justify GPTQ work.

## GPTQ gate (remains open)

The completed baseline records the model and dataset commits, environment,
token count, mean NLL, and PPL. Before starting GPTQ, define and pin a separate
calibration corpus and record calibration count, token count, group size,
damping, act-order, symmetry, seed, and bit widths. GPTQ results must compare
F0, F1/F2, F3/F4, and F5 on the same pinned WikiText-2 protocol; no synthetic
token result substitutes for this gate.
