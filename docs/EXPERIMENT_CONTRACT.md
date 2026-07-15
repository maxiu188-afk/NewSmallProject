# Experiment contract

Every result starts from a JSON configuration validated by
`scripts/validate_config.py`. The required fields are model identity/revision,
quantization mode and bit-widths, calibration and evaluation datasets/revisions,
seed, runtime environment, and upstream commit.

## Result labels

| Label | Meaning |
|---|---|
| `bf16` / `fp16` | Floating-point baseline; no low-bit execution claim |
| `fake_quant` | Quantize-dequantize simulation in floating-point operators |
| `packed_weight` | Weights stored in packed low-bit form; computation still unspecified |
| `int4_gemm` | Verified custom low-bit linear kernel is used |
| `int4_kv_cache` | Verified packed int4 KV cache and attention path are used |

`fake_quant` rows belong only in accuracy tables. `int4_gemm` and
`int4_kv_cache` require the exact CUDA build, GPU, shapes, and numerical
correctness result to be recorded before they appear in performance tables.
