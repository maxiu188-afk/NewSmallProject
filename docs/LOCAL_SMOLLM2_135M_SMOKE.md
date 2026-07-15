# SmolLM2-135M local pipeline smoke

## Purpose and scope

This is a local CPU correctness smoke for the configuration-driven LLaMA
adapter. It is not a language-model accuracy experiment, a calibration run, a
KV-cache result, or a low-bit deployment measurement. Synthetic token IDs are
used intentionally so that no dataset is downloaded or treated as evaluation
evidence.

## Reproducible model input

| Item | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Snapshot revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Weight artifact | `model.safetensors`, 269,060,552 bytes |
| Local configuration | `configs/pipeline/smollm2_135m_local.json` |
| Runtime | CPU, float32, one synthetic batch of 1 x 32 tokens |
| Packages | PyTorch 2.2.1, Transformers 4.40.1, NumPy 1.26.4 |

The local snapshot manifest is written under `results/model-prep/` and is
intentionally untracked. The immutable revision above is the tracked record.

## Command

macOS/Linux:

```bash
.venv-smollm/bin/python scripts/run_quarot_pipeline.py \
  configs/pipeline/smollm2_135m_local.json \
  --output results/pipeline-smollm2-135m-equivalence/result.json
```

Windows PowerShell uses the same configuration after an independently created
virtual environment:

```powershell
.venv-smollm\Scripts\python.exe scripts\run_quarot_pipeline.py `
  configs\pipeline\smollm2_135m_local.json `
  --output results\pipeline-smollm2-135m-equivalence\result.json
```

## Observed result

Run on the local Mac CPU after the model was loaded with `local_files_only`:

| Metric | Reference | Rotated candidate |
| --- | ---: | ---: |
| Mean NLL on synthetic tokens | 8.830812516 | 8.830814485 |
| Perplexity on synthetic tokens | 6841.843664 | 6841.857134 |
| First-logit mean absolute error | — | 3.8490e-05 |
| First-logit max absolute error | — | 8.4114e-04 |

The candidate uses seeded random residual rotation, V/O compensation, and the
online structured MLP Hadamard for `1536 = 12 x 128`. It
also detected SmolLM2's tied input/output embedding and copied the output head
before applying the distinct input and final-output transformations. Without
that step the first run had a max-logit error of about 53.55, so it was rejected
and is not retained as a valid result.

The remaining nonzero error is expected finite-precision accumulation across
the pretrained stack with a float32 QR-derived rotation; it is substantially
smaller than the model logits and is recorded rather than rounded away. A
structured exact Hadamard implementation and layerwise diagnostics are still
needed before treating this as stronger numerical-equivalence evidence.

## Explicitly not covered

- Q/K post-RoPE rotation;
- W4/A4, K/V-cache QDQ, calibration, or dataset perplexity;
- CUDA kernels, memory, latency, and throughput.
