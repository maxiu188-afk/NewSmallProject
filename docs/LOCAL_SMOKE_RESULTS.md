# Local smoke results

## Run record

- Date: 2026-07-15
- Platform: macOS arm64, Apple M5; no NVIDIA CUDA
- Python: 3.9.6
- Scope: pure-Python reference math only; no PyTorch, Transformers, model
  weights, upstream CUDA extension, calibration dataset, or language-model PPL
  was used.
- Command: `python3 -m unittest discover -v`
- Result: 14 tests passed.

## Framework-free one-token LLaMA-style equivalence smoke test

The deterministic toy block fuses RMSNorm scales into adjacent projections and
applies the same classes of rotations used by QuaRot: residual stream, Q/K
head-space Hadamard, V/O head-space Hadamard, MLP online Hadamard, and output
head rotation.  It is a transformation-identity check only, not a pretrained
model result.

| Metric | Observed maximum absolute error |
|---|---:|
| Q/K per-head attention score | 1.257674520083185143e-17 |
| Recovered hidden state | 2.220446049250313081e-16 |
| Final logits | 1.387778780781445676e-16 |

These errors are floating-point roundoff and pass the `1e-11` smoke-test gate.
The next required confirmation is the same test in a PyTorch LLaMA-shaped model
is recorded below.

## PyTorch/Transformers tiny random-model smoke test

An isolated `.venv-smoke` environment contains `torch==2.2.1`,
`transformers==4.38.0`, and `numpy<2`. It instantiated a two-layer random
LLaMA from `LlamaConfig` on CPU; it did not call `from_pretrained`, download a
tokenizer, use a dataset, or invoke CUDA/MPS.

The test checks RMSNorm fusion, residual rotation, the static V/O compensation
pair, MLP online Hadamard compensation, and LM-head equivalence through the
actual Transformers module graph. Q/K rotation after RoPE remains covered by
the framework-free test because it is injected into upstream attention internals.

| Metric | Observed maximum absolute error | Gate |
|---|---:|---:|
| Final logits | 1.1920928955078125e-07 | < 2e-05 |
| Recovered hidden states, all 3 saved tensors | 5.960464477539062e-07 | < 2e-05 |

The test passed. The slight FP32 differences are expected from reordered matrix
multiplications after fusion and rotation; they are not a quantization result.
