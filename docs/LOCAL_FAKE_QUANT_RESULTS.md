# Local fake-quant smoke results

## Run record

- Date: 2026-07-15
- Model: deterministic, random, two-layer tiny LLaMA (`hidden_size=32`,
  `intermediate_size=64`, 4 heads)
- Runtime: CPU, PyTorch 2.2.1, Transformers 4.38.0
- Quantizer: symmetric QDQ, per-output-channel weights and per-token linear
  inputs
- Reference: the same unquantized random model's logits

## Results

| Case | Mean absolute logit error | Max absolute logit error |
|---|---:|---:|
| F0 FP32 | 0.000000 | 0.000000 |
| F1 naive W4 | 0.011345 | 0.059377 |
| F2 QuaRot W4 | 0.010765 | 0.054235 |
| F3 naive W4A4 | 0.016088 | 0.088863 |
| F4 QuaRot W4A4 | 0.016125 | 0.071641 |

The QDQ paths are therefore active and the naive/rotated branches are distinct.
On this single random initialization, F2 has lower mean and maximum error than
F1; F4 has a lower maximum but a slightly higher mean error than F3. These
numbers are **not** evidence of QuaRot's empirical accuracy benefit because
the test has no pretrained weights, calibration set, PPL, zero-shot task, GPTQ,
or KV-cache quantization. They are retained only as local implementation smoke
evidence.
