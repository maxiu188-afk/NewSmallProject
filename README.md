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
  `train` calibration contract. Its first 13B server run is active but not yet
  an accepted result; review the persistent-volume artifacts first.

The Llama-2 BF16 result is a reproducible full-precision text-evaluation
control; the RTN F3/F4 results remain floating-point QDQ rather than deployment
evidence. All W4A4KV4 evidence remains an algorithmic simulation rather than
packed KV storage or a CUDA attention kernel. See
[phase status](docs/PHASE_STATUS.md), the
[Llama-2-13B BF16 baseline](docs/LLAMA2_13B_BF16_BASELINE.md), the
[Llama-2-13B RTN W4A4 result](docs/LLAMA2_13B_RTN_W4A4_RESULTS.md), the
[active GPTQ session checklist](docs/RUNPOD_GPTQ_SESSION.md), and the
[SmolLM2 fake-quant record](docs/LOCAL_SMOLLM2_135M_FAKE_QUANT.md).

## Layout

- `repro/`: independent, portable PyTorch implementations;
- `configs/`: model, data, runtime, rotation, and quantization configurations;
- `scripts/`: portable entry points;
- `tests/`: local correctness tests;
- `docs/`: experiment scope, results, and environment notes;
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
