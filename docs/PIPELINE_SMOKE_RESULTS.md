# Portable pipeline smoke results

## Configuration-driven GQA LLaMA equivalence

The portable pipeline was run locally on CPU using the random configuration in
`configs/pipeline/synthetic_llama_smoke.json`:

- 4 decoder layers, hidden size 256, MLP size 512;
- 4 query heads and 2 KV heads (GQA);
- 2 synthetic batches of length 32;
- residual Hadamard rotation, V/O compensation, and MLP online Hadamard;
- no quantization.

| Metric | Reference | Rotated candidate |
|---|---:|---:|
| Mean NLL | 5.553445 | 5.553445 |
| Synthetic PPL | 258.125176 | 258.125240 |
| Max absolute logit error | — | 1.609325e-06 |

The PPL is not a language-quality metric because both model weights and token
inputs are synthetic. It demonstrates that the configuration-driven GQA path
runs and preserves the model function within FP32 tolerance.

The companion W4A4 configuration runs the same pipeline with QDQ enabled.
Its numerical error is an implementation smoke signal only; it is not reported
as a pretrained-model QuaRot accuracy result.
