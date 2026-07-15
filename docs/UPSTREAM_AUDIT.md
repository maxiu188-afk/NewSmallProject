# Upstream QuaRot audit

## Reference snapshot

- Repository: `https://github.com/spcl/QuaRot.git`
- Local commit: `5008669b08c1f11f9b64d52d16fddd47ca754c5a` (2024-11-26)
- Submodules currently uninitialized locally:
  - CUTLASS: `ffa34e70756b0bc744e1dfcc115b5a991a68f132`
  - fast-hadamard-transform: `4ea722e434e3d4f2a14522341959ebdbe62be2de`
  - nvbench: `d8dced8a64d9ce305add92fa6d274fd49b569b7e`

## Observed paths

| Path | Observed purpose | Reproduction treatment |
|---|---|---|
| `fake_quant/` | LLaMA-oriented rotation, GPTQ/RTN, QDQ activation/KV simulation, PPL and LM-eval | Behavioural reference only; results labelled fake quant |
| `quarot/functional`, `quarot/nn` | int4 packing, quantizer, Hadamard module, packed W4 linear | Inspect and validate against independent reference |
| `quarot/kernels/` | CUDA/CUTLASS GEMM and FlashInfer-style KV kernels | RunPod only, after a recorded compatibility build |
| `e2e/` | LLaMA checkpoint conversion and prefill/decode benchmarks | RunPod only; not evidence until kernel correctness passes |

## Compatibility findings to preserve

- `requirements.txt` pins PyTorch 2.2.1 and Transformers 4.38.0.
- `setup.py` builds a CUDA extension and explicitly targets `sm_75`, `sm_80`,
  and `sm_86`.
- The E2E path requires CUDA, FlashAttention 2, custom kernels, and contains
  shape/layout restrictions such as head dimension 128 and limited paged-batch
  support.
- The fake-quant README says LLaMA-2 only, while the CLI lists additional
  models. The actual GPTQ and K-cache paths include LLaMA-specific logic, so
  LLaMA-2 is the first supported reproduction target.

No compatibility patch is justified before Phase 4 records the original RunPod
build result.

For future server work, see `CUDA_PYTORCH_COMPATIBILITY_POLICY.md`. In
particular, this repository's `CUDAExtension` means that the server's `nvcc`
toolkit and the selected PyTorch CUDA wheel must be checked as a build pair.
