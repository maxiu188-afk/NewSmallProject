# QuaRot reproduction

Research-oriented reproduction of QuaRot, with algorithmic fake-quantization
checks kept separate from real CUDA deployment and performance work.

## Current validated scope

- Primitive and small LLaMA correctness checks;
- A configuration-driven LLaMA-family pipeline;
- Offline SmolLM2-135M equivalence and F5 fake-quant smokes, including
  residual, V/O, Q/K-after-RoPE, and `12 x 128` MLP structured Hadamard
  transforms;
- Local W4, W4A4, and sequential-cache W4A4KV4 QDQ comparisons against a
  naive baseline.
- A recorded RunPod A40 CUDA smoke: 16-bit rotation equivalence plus matched
  SmolLM2-135M synthetic-input W4A4KV4 naive/QuaRot runs.
- A pinned Llama-2-13B BF16 WikiText-2 F0 baseline (`PPL=5.0083` over 331,614
  next-token targets).
- A matched Llama-2-13B RTN W4A4 text gate: naive F3 PPL `8719.6775` and
  QuaRot F4 PPL `12.4606` (K/V remain 16-bit; this is floating-point QDQ).
- A configuration-driven GPTQ W4A4 path with a separately pinned WikiText-2
  `train` calibration contract. The completed QuaRot F4 GPTQ result is
  `PPL=5.8376` (same-run BF16 `5.0087`); the matched naive GPTQ F3 control is
  `PPL=8624.3509`. The component and calibration results are recorded in the
  [consolidated fake-quant record](docs/FAKE_QUANT_RESULTS.md).
- An owned packed-W4/A8 CUDA integer-accumulator path validated for the three
  Llama-2-13B linear shapes, plus one formally GPTQ-packed `q_proj` integrated
  through a fixed-token decoder-layer/logits gate on RTX 6000 Ada. The complete
  280-linear packed decoder subsequently passed all 40 layer-output and final
  logits comparisons against the independent packed oracle with maximum error
  `0.0`; its BF16-K/V generation smoke also produced finite logits.
- A faithful pinned-upstream Llama-2-13B deployment with all 280 decoder
  projections executing as packed W4A4 and the paged K/V cache at 4 bits. On
  RTX 6000 Ada it reduced model-resident allocated memory from 26.29 GB to
  7.18 GB, but was slower than the matched upstream FP16 backend.
- A completed paper-protocol-aligned Llama-2-7B single-decoder-block benchmark
  on RTX 6000 Ada. W4A4KV4 completed all 14 formal cases, accelerated the paired
  2048-token prefill points by 1.53--1.68x, and reached a 1.28x layer-e2e
  speedup at batch 16/context 4096; FP16 OOMed only at batch 64/sequence 2048.
- A completed Llama-2-13B vLLM W4A16 serving study on Isambard GH200. Relative
  to BF16, both packed W4A16 checkpoints cut ready GPU memory by 52.7%,
  improved request throughput by 1.37--1.54x, and reduced p50 E2E by
  26.9--35.2%. A dependent real-vLLM layer-0 diagnostic found shape-dependent
  0.96--1.05x speed and 74.2% fewer parameter bytes. The matched deployed
  WikiText-2 gate measured PPL 5.007820 for BF16, 5.289677 for unrotated W4A16,
  and 5.132755 for rotated W4A16; rotation recovered 55.67% of the unrotated
  quantization gap. A separately scoped packed SpinQuant W4A16 endpoint reached
  79.7554% on the frozen 3,270-example BoolQ protocol versus 80.5810% BF16. Its
  matched serving run improved request throughput by 1.537x/1.379x and reduced
  ready GPU memory by 52.2%/52.7% at concurrency 1/8, with a small TTFT
  trade-off.
- An accepted Llama-2-13B deployed-checkpoint W4AFP8 PPL study through vLLM on
  Isambard GH200. Over 331,614 targets, PPL was 5.007820 BF16, 5.136105
  unrotated W4AFP8, 5.248356 QuaRot-style W4AFP8, and 5.230155
  SpinQuant-transfer W4AFP8. A matched observer diagnostic found that MSE
  clipping reduced the QuaRot-minus-unrotated gap by 52.8622% but did not
  reverse the ranking; unrotated min/max remained best. The FP8-targeted
  SpinQuant endpoint reached 80.2752% BoolQ, 1.42x/1.385x BF16 request
  throughput, and 50.7% lower ready GPU memory at concurrency 1/8.
- A completed formal SGLang-versus-vLLM W4A16 serving study on one GH200. All
  12 concurrency 1/8 x three-paired-repetition cells passed. SGLang had 20.45%
  higher median request throughput at concurrency 1 and 1.07% at concurrency
  8, but p50 TTFT was 25.87% and 64.89% higher; no blanket backend winner is
  claimed.

The Llama-2 BF16 result is a reproducible full-precision text-evaluation
control; the RTN F3/F4 results remain floating-point QDQ rather than deployment
evidence. Those historic W4A4KV4 rows remain algorithmic simulations, while
the separate official-backend result uses packed weights and the upstream
4-bit CUDA K/V cache. See the [documentation index](docs/README.md) and current
[phase status](docs/PHASE_STATUS.md) first. Algorithmic accuracy evidence is
consolidated in the [fake-quant result](docs/FAKE_QUANT_RESULTS.md); faithful
upstream full-model and single-block deployment evidence is consolidated in
the [official QuaRot result](docs/OFFICIAL_QUAROT_RESULTS.md). The owned
packed-W4/A8 path remains documented in the
[W4A8 kernel record](docs/W4A8_CUDA_KERNEL_RESULTS.md). The practical serving
evidence is consolidated in the
[vLLM W4A16 result](docs/VLLM_W4A16_RESULTS.md); deployed W4AFP8 quality is in
the [W4AFP8 result](docs/W4AFP8_RESULTS.md), and the formal cross-backend study
is in the [serving-backend result](docs/SERVING_BACKEND_COMPARISON_RESULTS.md).
Completed runbooks and detailed source records remain available under
[`docs/archive/`](docs/archive/README.md).

## Layout

- `repro/`: independent, portable PyTorch implementations;
- `configs/`: model, data, runtime, rotation, and quantization configurations;
- `scripts/`: portable entry points;
- `tests/`: local correctness tests;
- `docs/`: current documentation index, results, runbooks, and archived notes;
- `QuaRot/`: local upstream reference checkout, deliberately ignored by Git.

## Upstream reference

The upstream checkout is not vendored here. Obtain it beside this repository
when needed, then pin it to the audited reference commit:

```bash
git clone --recurse-submodules https://github.com/spcl/QuaRot.git QuaRot
git -C QuaRot checkout 5008669b08c1f11f9b64d52d16fddd47ca754c5a
```

See [the reproduction plan](QUAROT_REPRODUCTION_PLAN.md) for the two-track
algorithm/deployment strategy and [the portable pipeline guide](docs/PORTABLE_PIPELINE.md)
for local commands.

## Real-deployment direction

The completed deployment study keeps two explicitly named routes:

- faithful reproduction of the pinned upstream QuaRot W4A4 CUDA/e2e backend;
- standard-layout offline QuaRot rotations followed by a low-bit format and
  OpenAI-compatible serving path supported by stable vLLM.

Stable vLLM does not currently support INT4-weight/INT8-activation W4A8 on
NVIDIA GPUs, so the first serving format is W4A16 GPTQ on Ada/Hopper. This is
QuaRot-style engineering and is not labelled as original QuaRot W4A4. See
[the real-deployment roadmap](docs/QUAROT_REAL_DEPLOYMENT_ROADMAP.md) and the
[Isambard vLLM W4A16 result](docs/VLLM_W4A16_RESULTS.md).
The later backend-aligned W4AFP8 route now has accepted packed-checkpoint PPL,
matched serving, and BoolQ evidence; see the
[W4AFP8 result](docs/W4AFP8_RESULTS.md).
