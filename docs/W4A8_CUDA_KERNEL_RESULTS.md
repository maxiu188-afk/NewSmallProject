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

## Phase-2 integration smoke

The first Phase-2 boundary has passed for a real cached model tensor:
`model.layers.0.self_attn.q_proj` from the pinned Llama-2-13B checkpoint was
packed to groupwise W4 (group size 128), executed through `W4A8Linear`, and
compared with an independent floating computation using exactly the same packed
W4 values and per-token A8 values.  For a fixed CUDA BF16 input of shape
`[1, 2, 5120]`, the output shape was `[1, 2, 5120]` and maximum absolute error
was `1.9073486328125e-06`.

The recovered `llama-q-proj-smoke.json` artifact has SHA-256
`9b0e199bb3da22c7c319ba4f91d913a2207242e986d247fadb8f91841b90105d`.
This is RTN-style W4 packing integration evidence only.  It is not a
GPTQ-transformed packed checkpoint, full-layer/model equivalence, PPL, KV4, or
performance result.  The next Phase-2 task remains exporting and integrating
the formal GPTQ-transformed weights.

## Formal F4 GPTQ W4 export

That export gate has now passed for one real, formally calibrated Llama
linear.  The full F4 pipeline applied the configured QuaRot transformations,
then ran symmetric W4 GPTQ with activation ordering, group/block size 128, and
1% damping over 128 fixed WikiText-2 calibration sequences of 2,048 tokens.
It quantized all 40 decoder layers and 280 linear modules before exporting
`model.layers.0.self_attn.attention.q_proj`.

The exported artifact keeps the GPTQ act-order input permutation rather than
silently repacking by original column order.  It contains a `(5120, 2560)`
row-major uint8 tensor with two signed-int4 values per byte, FP32 scales of
shape `(5120, 40)`, that permutation, and no full-model checkpoint.  The
recovered files were SHA-256 verified locally:

| Artifact | SHA-256 |
|---|---|
| `llama2-13b-f4-layer0-qproj.pt` | `573feb9812f7e002b53a22fbc993ba79e9f7822f76833630db882d548585c9a6` |
| `llama2-13b-f4-layer0-qproj.json` | `4588fa52bdabb836812c062a7e8a54ef1576759ac8893f362491664c1a86f0bc` |

The exported W4 values reconstruct the in-memory BF16 GPTQ linear with maximum
absolute error `9.927153587341309e-05`; the actual CUDA `W4A8Linear` output
matches the independent packed floating oracle to
`5.662441253662109e-07` maximum absolute error.  The distinction matters: the
first value includes casting the GPTQ FP32 scale product to the model's BF16
weight storage, while the second validates the owned packed deployment path.

This is a self-describing **single-linear** formal-GPTQ deployment artifact and
correctness result.  It is not yet a packed full-model checkpoint, transformer
layer/model equivalence or PPL result, KV4 cache result, or performance claim.
The next gate is to replace that selected Llama linear in a fixed-token layer
execution while preserving the recorded rotation and input permutation.

## Formal fixed-token Llama layer gate

That next gate passed on the RTX 6000 Ada RunPod environment. The test loaded
the pinned Llama-2-13B model, applied the same QuaRot transformation, and made
two sequential forwards over fixed `[1, 16]` token IDs:

1. `PackedW4A8ReferenceLinear` used the exported signed W4 values, FP32 group
   scales, act-order input permutation, and per-token A8 rule through an
   independent Torch int32 accumulation oracle.
2. `W4A8Linear` replaced only that oracle with the owned CUDA int32 kernel.

Both replacements explicitly return BF16 at the surrounding transformer module
boundary. This is required because the CUDA/oracle accumulation and scale
application are FP32 while the Llama decoder layer remains BF16. The captured
layer output (`[1, 16, 5120]`) and model logits (`[1, 16, 32000]`) were exactly
equal at the declared `atol=2e-4`, `rtol=1e-5` comparison: max and mean absolute
errors were both `0.0` for this fixed input.

The recovered artifacts were copied locally and SHA-256 verified:

| Artifact | SHA-256 |
|---|---|
| `phase2-w4a8-layer/preflight.json` | `30c41a837c3a28736d36de49f26d883591c97fb52bfdd135cb308ddfeb955af4` |
| `phase2-w4a8-layer/w4a8-kernel-smoke.json` | `df63c72490689ea96f86d83af2775a21195ed469c02f038511f4fb78f10cf65f` |
| `phase2-w4a8-layer/llama2-13b-f4-qproj-layer-smoke.json` | `406d16c805fcf95b5854a13701b1a58e8f47bffd3f7642d9942ba24dcf207c61` |

This passes only the selected-linear fixed-token integration boundary. All
other linears and K/V remain BF16. It does not validate a packed full model,
generation, PPL, KV4, latency, throughput, or memory saving.

## Isambard-AI portability status

The same Phase-2 gate is prepared for the `brics.u6rt` project on
Isambard-AI. Its ARM64 CUDA 12.8 wheel index provides PyTorch
`2.9.0+cu128`, rather than the RunPod runtime's `2.8.0+cu128`; the environment
also records Transformers `5.14.1`, Datasets `5.0.0`, CUDA compiler `12.6`,
and `ninja==1.11.1.4`. The exact Llama revision and WikiText-2 splits were
verified from the local cache with outbound Hub access disabled.

The original jobs `5732906` and `5735242` stopped before CUDA because the
Slurm spool copy broke source-relative project-root discovery. Job `5739260`
then reached `kernel_smoke` but failed because the system GCC was too old for
PyTorch 2.9. The explicit GCC 13.2 repair ran as job `5742443` on 2026-07-22
and completed with exit code `0:0` on one GH200. It produced exact int32
accumulators, maximum scaled FP32 error `3.814697265625e-06`, and `0.0`
maximum/mean errors for both the fixed-token decoder-layer output and final
logits. This completes the selected-`q_proj` Isambard portability gate only;
all other linears and K/V remained BF16.

## Full-decoder gate preparation

The next gate is implemented locally but has no GPU result. It streams all 280
decoder-linear GPTQ captures to independent checksum-protected tensor shards,
publishes a complete manifest only after the expected set is present, and can
install those exact tensors through either the independent packed oracle or
the owned CUDA W4A8 module. The planned numerical gate compares every decoder
layer output and final logits before a short BF16-K/V generation smoke.

This preparation does not upgrade the selected-linear result into a full-model
claim. See `PHASE2_GPTQ_W4A8_FULL_MODEL_RUNBOOK.md` for the Isambard-primary
execution contract and RunPod fallback commands.

## Full-decoder RunPod correctness result

The prepared gate then passed on the RTX 6000 Ada environment at project
revision `5aa871341000306fb2aa78afc1a74c4ee1600aa7`. The sharded checkpoint
contains exactly 280 checksum-indexed decoder linears. Its manifest SHA-256 is
`caa12e485087e4bc5630c950e5b96041ba90e770cc40cdd7ae5defba64641e33`.

For fixed input IDs of shape `[1,16]`, the owned CUDA path and independent
packed oracle agreed at every one of the 40 decoder-layer boundaries and at
the final `[1,16,32000]` logits. Every maximum and mean absolute error was
`0.0`. A subsequent eight-token greedy smoke kept its BF16 K/V cache and had
finite logits. The recovered result JSON has SHA-256
`0a848e22ff1881ecd1ee76d829e3c39d43e2f9224892cdd1e61ada6be9b208d8`.

This establishes fixed-workload execution correctness for a complete W4A8
decoder with BF16 embedding, `lm_head`, and K/V. It does not establish PPL,
KV4, speed, throughput, or memory savings. The next permitted work is the
matched BF16/W4A8 measurement protocol in `W4A8_PERFORMANCE_RUNBOOK.md`.
