# Windows CUDA F5 smoke

## Environment

- Date: 2026-07-16
- Platform: Windows; NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB)
- Driver: 610.62
- Python: 3.11.8
- PyTorch: `2.2.1+cu121`; `torch.version.cuda == 12.1`
- Transformers: `4.38.0`

The isolated environment is `.venv-smoke`. It is local state and remains
ignored by Git.

## Command

```powershell
.venv-smoke\Scripts\python.exe scripts\run_quarot_pipeline.py `
  configs\pipeline\synthetic_llama_w4a4kv4_cuda_smoke.json `
  --output results\pipeline-synthetic-w4a4kv4-cuda-smoke\result.json
```

## Result

The run completed on `cuda` using a random four-layer GQA LLaMA and sequential
cached decoding. Q/K post-RoPE Hadamard, W4/A4/K4/V4 QDQ, and the cache path
were all active. The reported mean and maximum logit errors were `0.1150224`
and `0.5998298`; these are fake-quantization smoke metrics on synthetic token
IDs, not language-quality, deployment, memory, or throughput results.

The associated PyTorch/Transformers test group passed all nine tests. PyTorch
reported that this wheel lacks FlashAttention; no FlashAttention or performance
claim is made by this smoke.
