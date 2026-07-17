# Llama-2-13B WikiText-2 BF16 baseline

## Result

The first formal text-evaluation record completed on 2026-07-17. It is the
F0 full-precision control for subsequent RTN and GPTQ fake-quant experiments;
it is not a low-bit, QuaRot, GPTQ, KV-cache, memory, or throughput result.

| Item | Value |
|---|---|
| Model | `meta-llama/Llama-2-13b-hf` |
| Model revision | `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test` |
| Dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Runtime | CUDA on one NVIDIA A40 (44.42 GiB visible), PyTorch `2.11.0+cu128`, BF16 |
| Evaluation protocol | 162 non-overlapping sequences of length 2048; 331,614 next-token targets |
| Mean NLL | `1.6111028514668504` |
| Perplexity | `5.008331629066348` |

All quantization precisions were 16 bits, rotation was disabled, and KV-cache
simulation was disabled. The result JSON therefore reports identical
`reference` and `candidate` fields and zero logit error by construction: the
F0 code path reuses the reference evaluation rather than performing a second
independent candidate forward pass.

## Provenance and retained files

The server generated, and the project owner downloaded, the following ignored
artifacts under `results/`:

- `runpod-preflight/preflight-llama13b-baseline.json`;
- `model-prep/llama2-13b-manifest.json` and
  `model-prep/llama2-13b-model-pinned.json`;
- `data-prep/wikitext2-manifest.json` and
  `configs/llama2-13b-wikitext2-bf16-pinned.json`;
- `llama2-13b-wikitext2-bf16/result.json`.

Before server shutdown, every result JSON parsed successfully; the model
manifest listed 19 files and all 19 matched their recorded byte sizes. The
model cache occupied about 49 GiB on the persistent volume. The downloaded
result files had SHA-256 hashes matching the server copies.

## Interpretation and boundary

This establishes a stable internal control: later rows must use the same
model/data revisions, evaluation split, context length, and scorer before a
PPL difference is attributed to quantization. The absolute PPL must not be
compared directly with a paper or another evaluator unless its tokenization,
document-boundary handling, context/stride, and treatment of the trailing
tokens agree. The current evaluator concatenates text rows without added
special tokens, forms non-overlapping 2048-token blocks, and drops the final
incomplete block.

The next experiment remains open: define a separate pinned calibration corpus
and implement a paper-aligned GPTQ path. Record group size, damping, act-order,
symmetry, calibration token count, and seed. The current RTN-style QDQ code
must not be labelled GPTQ.

## Download and authentication note

The server's `hf-xet 1.5.2` failed during the first authenticated model fetch
with `DataHashHexParseError`. The successful preparation used
`HF_HUB_DISABLE_XET=1`. Also, the current `prepare_hf_model.py` and
`run_quarot_pipeline.py` explicitly use the ignored project cache
`.cache/huggingface`; an external `HF_HOME` is overwritten. Authenticate into
that cache for the current revision, and keep the token and cache out of Git.
