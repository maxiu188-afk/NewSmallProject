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

## Windows CUDA F5 algorithm smoke

On 2026-07-16, the new F5 configuration was run on the local Windows NVIDIA
GPU before any server experiment. The isolated `.venv-smoke` environment used
Python 3.11.8, PyTorch `2.2.1+cu121`, Transformers `4.38.0`, and an RTX 3070
Ti Laptop GPU (8 GB). The command used
`configs/pipeline/synthetic_llama_w4a4kv4_cuda_smoke.json` and the same random
four-layer GQA LLaMA shape as the CPU pipeline smoke.

The run completed with `device=cuda` and `kv_cache_simulated=true`. It applied
residual and V/O rotations, Q/K per-head Hadamard immediately after RoPE,
power-of-two MLP online Hadamard, W4/A4/K4/V4 QDQ, and sequential cached
decoding.

| Metric | Reference | W4A4KV4 candidate |
|---|---:|---:|
| Mean NLL on synthetic tokens | 5.553445 | 5.528105 |
| Synthetic PPL | 258.125367 | 251.666604 |
| Mean absolute logit error | — | 0.115022 |
| Max absolute logit error | — | 0.599830 |

All nine PyTorch/Transformers smoke tests passed in the same environment.
These values establish that the local CUDA F5 code path runs; synthetic token
IDs mean they are not language-quality, pretrained accuracy, storage, memory,
or throughput evidence. The complete environment and command record is in
`WINDOWS_CUDA_SMOKE.md`.
