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
boundary are recorded in
[LLAMA2_13B_BF16_BASELINE.md](../fake_quant/LLAMA2_13B_BF16_BASELINE.md).

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

## Completed RTN W4A4 gate

The matched pair completed on 2026-07-17. The comparison gate passed with
identical pinned model/data/evaluation fields: F3 naive RTN W4A4 scored PPL
`8719.677539`, while F4 QuaRot RTN W4A4 scored `12.460633`. The matched BF16
reference in these runs scored `5.008573`. Full provenance, numerical
boundaries, and the saved result locations are in
[LLAMA2_13B_RTN_W4A4_RESULTS.md](../fake_quant/LLAMA2_13B_RTN_W4A4_RESULTS.md).

## GPTQ implementation and gate

The GPTQ implementation is now configuration-driven, but its 13B result is
not yet claimed. It uses a separately pinned calibration source and processes
the LLaMA decoder one layer at a time. For each `nn.Linear`, it accumulates the
input Hessian on calibration tokens, applies damping and activation ordering,
then quantizes W4 columns while propagating the quantization error through the
inverse Hessian. It remains a floating-point fake-quant weight transform: it
does not pack weights or provide a custom CUDA kernel.

Both initial GPTQ controls use `Salesforce/wikitext`,
`wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`,
but keep its roles separate:

- Evaluation is the existing `test` split (162 non-overlapping sequences of
  length 2048).
- Calibration is the `train` split, consumed deterministically in source order
  as 128 sequences of length 2048 (262,144 input tokens). It has a row ceiling
  of 100,000 only to make the source bound explicit; `max_batches: 128` is the
  actual calibration limit.
- Both candidates use symmetric W4, group size 128, 1% Hessian damping,
  block size 128, and activation-ordering enabled. Their evaluation stays A4
  with K/V at 16-bit, so a change in the result is attributable to GPTQ weight
  fitting and/or QuaRot rather than KV-cache quantization.

Run the matched naive and QuaRot candidates only after the GPU end-to-end smoke
test completes:

```bash
python3 scripts/run_quarot_pipeline.py \
  configs/pipeline/llama2_13b_wikitext2_naive_w4a4_gptq.json \
  --output results/llama2-13b-wikitext2-gptq-w4a4/naive-f3-gptq.json

python3 scripts/run_quarot_pipeline.py \
  configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_gptq.json \
  --output results/llama2-13b-wikitext2-gptq-w4a4/quarot-f4-gptq.json
```

The result JSON records both data roles, the actual number of calibration
sequences/tokens, all GPTQ hyperparameters, and each quantized linear layer's
estimated loss. GPTQ results must compare F0, F3/F4 RTN, and F3/F4 GPTQ on the
same pinned WikiText-2 protocol; no synthetic-token result substitutes for
this gate.

## Completed CUDA 12.4 F4 GPTQ run

The initial QuaRot F4 GPTQ command completed on an RTX 6000 Ada using an
independent container-disk Python 3.12 environment with PyTorch `2.6.0+cu124`
and the persistent `/workspace/NewSmallProject/.cache/huggingface` cache. Its
same-run BF16 reference scored PPL `5.008716`; the fully calibrated F4 GPTQ
candidate scored `5.837575`. The result JSON records 128 train calibration
sequences, all 40 decoder layers, and 280 quantized linear layers; raw files
were copied and checksum-verified before server shutdown.

This completes the F4 GPTQ accuracy gate and shows a clear improvement over
F4 RTN. The next experiment is the matched naive GPTQ F3 control, not a repeat
of F4; only that pair can isolate QuaRot's effect under GPTQ. See
`LLAMA2_13B_GPTQ_W4A4_RESULTS.md` for the result boundary and provenance.

## Completed component and calibration study

The follow-up server session completed the matched naive F3 GPTQ control, a
cumulative QuaRot component ablation, and the matched 32/64/128
calibration-size study. It retained the pinned model/data/runtime contract and
W4A4, K/V-16 GPTQ settings. The raw server JSON/log files were recovered and
SHA-256 checked. See `LLAMA2_13B_GPTQ_ABLATION_CALIBRATION_RESULTS.md` for the
result table and its interpretation boundary.
