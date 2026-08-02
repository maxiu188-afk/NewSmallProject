# QuaRot real-deployment roadmap

## Objective and naming

The completed deliverable contains two separate QuaRot evidence tracks. The
official-backend track reproduces upstream QuaRot W4A4 execution. The serving
track combines only offline-fusible QuaRot rotations with a low-bit format
natively supported by stable vLLM; it is reported as **QuaRot-style**, not as
original QuaRot W4A4. A third, planned W4AFP8 route will compare QuaRot-style
and SpinQuant offline rotations under one GH200 deployment protocol.

The existing owned W4A8 correctness kernel remains useful evidence that all
280 Llama-2-13B decoder linears can execute from packed W4. Its performance
smoke reduced peak allocated memory from about 26.2 GB to 7.4--7.5 GB but was
slower than BF16, so it is no longer the default deployment backend. No new
full CUDA kernel was developed for the completed routes. SpinQuant is excluded
from those completed QuaRot results, but is included as a separately named
method in the planned W4AFP8 comparison.

## Feasibility decisions from the current repository

### Route A: official QuaRot backend

This route is feasible on an x86-64 NVIDIA development image, with a bounded
compatibility step. The pinned upstream revision is
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`. Its checkpoint converter declares
Llama-2 7B, 13B, and 70B; its native path contains packed W4A4 linears, int32
accumulation, a paged int4 KV cache, and custom Llama model classes.

Current prerequisites and risks are concrete:

- CUTLASS, fast-hadamard-transform, and nvbench submodules are not initialized
  in the local checkout.
- Upstream pins PyTorch 2.2.1 and Transformers 4.38.0.
- `setup.py` emits only `sm_75`, `sm_80`, and `sm_86`; RTX 6000 Ada is `sm_89`.
- The model code imports the Transformers 4.38 `LlamaFlashAttention2` API and
  requires FlashAttention 2 for prefill.
- KV decode requires `head_dim == 128`, and the cache has limited page-layout
  support.
- Isambard GH200 is ARM64 and its prepared environment already showed that old
  PyTorch CUDA wheels are not generally available there. It is not the first
  platform for this legacy backend.

The first GPU target is therefore a RunPod x86-64 NVIDIA CUDA development
image. We first record an unmodified build. If it fails only because Ada is
absent from the architecture list, the allowed patch is limited to adding
`sm_89`; algorithm kernels and layouts remain unchanged.

### Route B: offline QuaRot rotation plus vLLM

This route is feasible for serving, but not with NVIDIA GPU W4A8 under the
current stable vLLM support matrix. Stable vLLM lists INT8 W4A8 as Arm-CPU-only,
while GPTQ/AWQ/Marlin W4A16 is supported on Ada and Hopper. Consequently:

- GPU W4A8 is a rejected first target unless stable vLLM adds support later.
- We will not implement an out-of-tree W4A8 plugin or another CUDA kernel.
- The first NVIDIA format is group-128 GPTQ W4A16, using vLLM/LLM Compressor
  output rather than the repository's owned checkpoint format.

Both available GPU classes are plausible for this route: RTX 6000 Ada and
GH200 satisfy the vLLM hardware/quantization matrix. Route B now selects
Isambard GH200 as its formal platform after the RTX 6000 Ada RunPod session
completed the paper-aligned official QuaRot single-block result. A read-only
check on 2026-07-21 confirmed `aarch64`, glibc 2.38, and `/usr/bin/apptainer`
plus `/usr/bin/singularity`; the selected-linear W4A8 CUDA gate then passed on
GH200 as job `5742443`. These facts establish usable CUDA allocation and
container feasibility.

The 2026-07-22 login-node preparation froze the official vLLM
`0.25.1+cu129` aarch64 wheel with PyTorch `2.11.0+cu129`, plus a separate LLM
Compressor `0.12.0` environment. The serving and quantizer environments remain
separate because they pin compressed-tensors 0.17.0 and 0.17.1 respectively.
The full evidence chain has since passed on GH200: tiny checkpoint execution,
Llama-2-13B export and fixed-token inference, deployed-checkpoint WikiText-2
PPL, service smoke, matched serving, and a dependent real-layer diagnostic. The
two complete W4A16 checkpoints each contain 280 packed decoder linears.

The serving checkpoint may use the residual Hadamard and the paired per-head
V/O Hadamard because these can be absorbed into standard Llama weights. It
must not contain online MLP Hadamards, post-RoPE Q/K wrappers, custom attention
classes, or a custom KV cache. The first implementation in
`repro/offline_llama_rotation.py` preserves standard Transformers Llama module
types, fuses RMSNorm scales by setting their stored weights to one, and verifies
save/reload equivalence on a tiny random model before any quantization.

Current references:

- vLLM stable quantization matrix: <https://docs.vllm.ai/en/stable/features/quantization/>
- vLLM NVIDIA/aarch64 installation guidance: <https://docs.vllm.ai/en/stable/getting_started/installation/gpu/>
- vLLM W4A8 recipe and limitations: <https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int8_w4a8/>
- LLM Compressor W4A16 example: <https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/>
- vLLM OpenAI-compatible server: <https://docs.vllm.ai/en/stable/serving/openai_compatible_server/>

### Route C: joint hardware-aligned W4AFP8 extension

Route C is planned but not started. It begins only after the current SpinQuant
`no_had` W4A8 fake-quant dependency chain is complete and accepted. It will
export four matched Llama-2-13B cases: BF16, unrotated W4AFP8, QuaRot-style
W4AFP8, and SpinQuant W4AFP8. All rotations remain offline; KV cache and
non-linear components remain 16-bit for the first study.

This route must prove the selected Hopper W4AFP8 kernel, then measure both
deployed-checkpoint WikiText-2 PPL and full-model service performance. Existing
INT8 W4A8 fake-quant numbers cannot be used as FP8 accuracy evidence. The
complete dependency, export, quality, serving, and acceptance gates are frozen
in [`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md).

## Execution gates

### A. Official-backend gates

1. **Source gate:** run `scripts/audit_upstream_quarot_backend.py`; verify the
   pinned revision, initialize exact submodules, and retain their SHAs.
2. **Build gate:** use a dedicated legacy x86-64 CUDA environment, attempt the
   unmodified editable build, then apply at most the reviewed `sm_89`
   compatibility change.
3. **Primitive correctness:** compare pack/unpack, activation quantization,
   int4 GEMM int32 accumulators, and scaled outputs with Torch references.
4. **KV correctness:** compare int4 cache initialize/append/decode with an
   unpacked FP16 reference at a short context and one page boundary.
5. **Small e2e:** run `e2e/benchmark_layer.py` on one Llama-2-7B decoder layer
   with fixed tensors; require finite outputs and matched FP16/int4 shape.
6. **Full-model correctness:** convert Llama-2-7B first, compare fixed-token
   logits/generation with the fake-quant reference, then run a small PPL slice.
7. **Performance:** only after correctness, measure linear, KV decode, layer,
   prefill, decode, and e2e with warm-up and repeated CUDA-synchronized samples.

The raw upstream `e2e/benchmark.py` uses zero warm-up and one timed iteration.
It may be retained as a reproduction artifact, but presentation numbers must
come from the matched repeated protocol.

### B. vLLM-serving gates

1. **Offline transform smoke:** run
   `scripts/run_offline_llama_rotation_smoke.py`; direct and save/reload logits
   must match the original tiny Llama within `2e-5`, and every decoder
   projection must remain `nn.Linear`.
2. **Quantizer smoke:** export three tiny/random Llama checkpoints: BF16,
   unrotated W4A16, and offline-rotated W4A16. Validate compressed-tensors
   metadata and reload each with the same stable vLLM release.
3. **vLLM offline inference:** compare fixed-token next-token logits and greedy
   token IDs. Rotation error is measured before quantization; quantization error
   is measured against each model's BF16 source.
4. **Service smoke:** launch `vllm serve`, check `/health` and `/v1/models`, and
   send one deterministic completion request.
5. **Pretrained scale-up:** repeat on the smallest approved pretrained Llama
   checkpoint before Llama-2-13B. Use identical calibration samples for the
   rotated and unrotated W4A16 pair.
6. **Matched comparison:** BF16 original, unrotated W4A16, and offline-rotated
   W4A16 use the same prompts, concurrency, input/output lengths, vLLM version,
   GPU, and server flags. Record quality, TTFT, TPOT, request throughput, token
   throughput, and peak memory.

### C. W4AFP8 joint-deployment gates

1. **Dependency gate:** accept jobs `5850956` and `5850958`; do not implement,
   export, or submit W4AFP8 work before that point.
2. **Runtime gate:** validate metadata, all 280 packed linears, alignment, FP8
   scales, and the actual selected Hopper W4AFP8 kernel in fresh processes.
3. **Quality gate:** measure the same retained WikiText-2 PPL from every
   deployed checkpoint; fake-quant PPL is not a substitute.
4. **Performance gate:** run BF16 and all three W4AFP8 cases with the matched
   full-model service protocol and report latency, throughput, memory, failures,
   and recovery.
5. **Claim gate:** keep QuaRot-style and SpinQuant results separate and call a
   speedup only after verified W4AFP8 kernel execution.

## Completed deployment results

Route A's pinned source/build/primitive/KV gates, complete Llama-2-13B export,
full-checkpoint generation smoke, and matched FP16/W4A4KV4 full-model extension
have completed on RTX 6000 Ada. That result reduces model-resident allocated
memory from 26.29 GB to 7.18 GB but is slower at batch 1; it is not the
single-transformer-block performance protocol reported by the paper.

The pinned upstream Llama-2-7B single-block matrix has completed on the same
RunPod RTX 6000 Ada platform. W4A4KV4 completed all 14 cases; FP16 completed 13
and retained a capacity OOM at batch 64/sequence 2048. Paired 2048-token
prefill speedup was 1.53--1.68x, while batch-16/context-4096 layer-e2e reached
1.28x. The Ada result is protocol-aligned rather than numerically exact because
the paper used RTX 3090. See `OFFICIAL_QUAROT_RESULTS.md`.

Route B has completed on Isambard GH200. In the primary matched serving
benchmark, both W4A16 checkpoints reduced ready GPU memory from 34,099 to
16,125 MiB. Relative to BF16, request throughput improved by 1.53--1.54x at
concurrency 1 and 1.37x at concurrency 8; p50 E2E fell by 26.9--35.2%.
Unrotated and offline-rotated W4A16 had effectively the same deployment
performance.

The matched deployed-checkpoint quality job scored 331,614 WikiText-2 targets
through vLLM. PPL was 5.007820 for BF16, 5.289677 for unrotated W4A16, and
5.132755 for rotated W4A16. Offline rotation therefore reduced PPL by 2.97%
relative to unrotated W4A16 and recovered 55.67% of its PPL gap to BF16 without
materially separating serving performance.

The dependent eager-mode layer-0 diagnostic found a 74.2% parameter-byte
reduction, while median speed ranged from 0.96x to 1.05x versus BF16 depending
on shape. It confirms execution of the real packed vLLM layer but does not
replace the compiled full-model result.

The remaining Route B quality boundary is downstream-task or broader generation
evaluation; deployed-checkpoint WikiText-2 PPL is complete. Results from Routes
A and B remain separately named and must not be merged into one precision or
performance label. See `VLLM_W4A16_RESULTS.md`.
