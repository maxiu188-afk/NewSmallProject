# W4A8 CUDA kernel correctness results

## Result

The owned Phase-1 W4A8 CUDA correctness kernel passed on the recorded RTX 6000
Ada environment.  It accepts per-token signed A8 values and row-major packed
signed W4 weights, then returns one `int32` accumulator per token, output row,
and weight group.  The kernel source is owned by this repository; the ignored
upstream QuaRot tree was not built or invoked.

The small scaled-output smoke passed with exact integer accumulators and a
maximum FP32 scaled-output absolute error of `3.814697265625e-06`.  Its shape
was 3 tokens x 16 output features x 128 input features, with group size 32.

The Llama-2-13B shape matrix then passed exact elementwise int32 comparison at
one token and group size 128:

| Linear shape `(out, in)` | Group accumulators | Exact int32 match |
|---|---:|---:|
| `(5120, 5120)` | `[1, 5120, 40]` | yes |
| `(13824, 5120)` | `[1, 13824, 40]` | yes |
| `(5120, 13824)` | `[1, 5120, 108]` | yes |

## Recorded runtime

- GPU: NVIDIA RTX 6000 Ada Generation, compute capability `8.9`, 49,140 MiB;
- driver: `570.124.06`;
- CUDA compiler: `12.8.93`;
- PyTorch: `2.8.0+cu128` (`torch.version.cuda == 12.8`);
- Python: `3.12.3`;
- kernel architecture target: `sm_89`;
- shape-matrix implementation revision:
  `d2a6b2db4ddb9d843ae6e25b5775631f53d9faa9`.

The container-disk environment
`/opt/newsmallproject/venvs/w4a8-cu128` uses the server's matching PyTorch CUDA
runtime and adds `ninja==1.11.1.4` plus Transformers `5.14.1`.  Project code,
the 13B model cache, and result artifacts remain on `/workspace`; a virtual
environment is not retained there.  The versioned CUDA toolkit is discovered
at `/usr/local/cuda/bin/nvcc`, rather than assumed to be on `PATH`.

## Artifact verification

Raw generated artifacts remain ignored by Git.  They were copied from the
persistent server volume and SHA-256 verified locally:

| Artifact | SHA-256 |
|---|---|
| `preflight.json` | `6bb0b3d8113be79174b2bb523f7aba8a86559e5c87b78d5cf731e22d8b5fd47b` |
| `smoke.json` | `d2b2f07dd81ec824f422f4e204c1928b37262dda42ba3c823a9407627e4bbc28` |
| `llama-shape-matrix.json` | `8e934bd998cbca52770ba7e2bd40952613458bd9cabd8882b897908b84404444` |
| `container-env-smoke.json` | `833342406c729a5d036427a4d3d7bbf2f3ff28e5233df94e5f3a10b61ce1d8e6` |
| `container-pip-freeze.txt` | `7689a6a20d9ca00a39c84fa70a93fc5a63ab1550d97eb7b3fdb772b00ed158c9` |

## Scope and next gate

This passes the `int4_gemm` numerical-correctness gate for the listed tensor
shapes and the recorded GPU/toolchain.  It does **not** yet demonstrate a
packed GPTQ checkpoint, transformer-layer integration, language-model PPL,
KV4 cache, latency, throughput, or memory savings.

The next work is Phase 2: export a GPTQ-transformed Llama linear tensor to the
checksummed packed-W4 format, replace selected Llama linears with the verified
W4A8 path, and compare fixed-token layer outputs against the existing GPTQ
floating reference before adding KV4.
