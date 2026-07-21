# W4A8 performance benchmark runbook

## Claim boundary

This benchmark is permitted because the complete 280-linear W4A8 decoder path
has passed the fixed-token layer/logits and BF16-K/V generation correctness
gate. The measured candidate is:

- every decoder `nn.Linear`: packed GPTQ W4 with per-token symmetric A8;
- embedding and `lm_head`: BF16;
- K/V cache: BF16.

The matched baseline loads the same pinned Llama-2-13B model and applies the
same QuaRot transformation, but leaves all decoder linears in BF16. The packed
oracle is never called inside a timed region. These results cannot be labelled
KV4, W4A4, or W4A8KV4 performance.

## Timing protocol

Two checked workload configurations are provided:

- `configs/deployment/llama2_13b_w4a8_benchmark_smoke.json` is a short
  execution and result-schema gate;
- `configs/deployment/llama2_13b_w4a8_benchmark.json` is the formal workload
  grid frozen before the first performance run.

Both run BF16 first, release it, then load W4A8. They record every raw sample
and summarize CUDA-event latency, synchronized wall latency, median, p95,
standard deviation, work items per second, peak allocated memory, and peak
reserved memory. Each timed sample is CUDA-synchronized on both sides.

The sections have deliberately distinct boundaries:

- `linear`: direct representative decoder-linear calls for `(5120,5120)`,
  `(13824,5120)`, and `(5120,13824)`; the surrounding online Hadamard is
  excluded from both modes;
- `prefill`: one complete model forward with `use_cache=False`, reported in
  prompt tokens/s;
- `decode`: build a BF16 KV cache at the requested context length outside the
  timed region, then time exactly one new-token forward;
- `generation`: time the complete greedy prefill and cached decode loop,
  reported in generated tokens/s.

Model loading, QuaRot transformation, checkpoint checksum validation, and CUDA
extension compilation are not timed. Static live model memory is recorded only
after freed BF16 replacement allocations have been returned to the CUDA
allocator. Timing begins only after configured warm-up iterations.

## Local validation

This checks the workload schema and links it to the recovered correctness
result and checkpoint manifest without requiring their 280 weight shards:

```bash
python scripts/run_llama_w4a8_benchmark.py \
  --config configs/deployment/llama2_13b_w4a8_benchmark_smoke.json \
  --validate-only

python scripts/run_llama_w4a8_benchmark.py \
  --config configs/deployment/llama2_13b_w4a8_benchmark.json \
  --validate-only
```

Local macOS validation is configuration and result-aggregation evidence only.
Do not run the benchmark on CPU or MPS.

## Isambard primary run

Isambard remains the primary formal platform. Submit only after its own
280-linear checkpoint and `isambard-full-model-smoke.json` have passed and the
service is healthy:

```bash
cd "$HOME/NewSmallProject"
sbatch --test-only scripts/run_isambard_w4a8_benchmark.sbatch
sbatch scripts/run_isambard_w4a8_benchmark.sbatch
```

The requested 24-hour wall time must be accepted by `--test-only`; it is not a
current scheduler guarantee. The batch job rejects a dirty checkout, validates
the linked Isambard correctness result, runs the smoke grid first, then the
formal grid. It emits:

```text
W4A8_BENCHMARK_STAGE=module_load
W4A8_BENCHMARK_STAGE=inputs
W4A8_BENCHMARK_STAGE=source_manifest
W4A8_BENCHMARK_STAGE=preflight
W4A8_BENCHMARK_STAGE=smoke
W4A8_BENCHMARK_STAGE=formal
W4A8_BENCHMARK_STAGE=assert_result
```

Success requires `ISAMBARD_PHASE3_W4A8_BENCHMARK_PASSED` and a complete JSON
with both modes and matched comparison rows.

## RunPod fallback

Use the existing persistent checkpoint and recovered RunPod correctness result
with the same checked configs:

```bash
python scripts/run_llama_w4a8_benchmark.py \
  --config configs/deployment/llama2_13b_w4a8_benchmark_smoke.json \
  --output results/phase3-w4a8-benchmark/runpod-benchmark-smoke.json

python scripts/run_llama_w4a8_benchmark.py \
  --config configs/deployment/llama2_13b_w4a8_benchmark.json \
  --output results/phase3-w4a8-benchmark/runpod-benchmark.json
```

Run the formal grid only after the smoke result has `status: complete`. A
correctness-first kernel may be slower than BF16; record that result before any
optimization. Every later kernel optimization invalidates the previous timing
row until the numerical correctness gates are rerun.
