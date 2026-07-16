# QuaRot reproduction

Research-oriented reproduction of QuaRot, with algorithmic fake-quantization
checks kept separate from real CUDA deployment and performance work.

## Current local scope

- Primitive and small LLaMA correctness checks;
- A configuration-driven LLaMA-family pipeline;
- Offline SmolLM2-135M equivalence and F5 fake-quant smokes, including
  residual, V/O, Q/K-after-RoPE, and `12 x 128` MLP structured Hadamard
  transforms;
- Local W4, W4A4, and sequential-cache W4A4KV4 QDQ comparisons against a
  naive baseline.

The current results are code-path checks on fixed synthetic inputs. They are
not pretrained text-evaluation claims or real low-bit inference benchmarks.
The W4A4KV4 path remains an algorithmic QDQ simulation, not packed KV storage
or a CUDA attention kernel. See [phase status](docs/PHASE_STATUS.md) and the
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
