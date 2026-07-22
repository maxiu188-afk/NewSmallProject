# Official QuaRot W4A4 backend result

## Outcome

The pinned upstream QuaRot backend now loads and executes a complete exported
Llama-2-13B checkpoint on one NVIDIA RTX 6000 Ada. This is the official-backend
route: all 280 decoder projections use upstream packed 4-bit weights and the
upstream integer-accumulator CUDA kernel, activations are quantized to 4 bits,
and the paged cache stores K/V at 4 bits. It is therefore real W4A4KV4
execution, not the repository's earlier floating QDQ simulation and not the
QuaRot-style vLLM route.

The result is a capacity win, not a speedup on this GPU. Against the matched
upstream QuaRot FP16 model/cache implementation, W4A4KV4 reduced model-resident
allocated GPU memory from 26.29 GB to 7.18 GB, but was slower in every measured
workload.

This is a full-model engineering extension. The QuaRot paper's performance
figures and tables time one transformer block, so this result must not be
presented as a reproduction of those numbers. The subsequent paper-aligned
Llama-2-7B single-block RTX 6000 Ada run is recorded separately in
`OFFICIAL_QUAROT_SINGLE_BLOCK_RESULTS.md`.

## Provenance and compatibility scope

- Project revision used for the formal benchmark: `e23d4cba0659b0f2b88dcf332f102e4c8500c786`
- Upstream QuaRot revision: `5008669b08c1f11f9b64d52d16fddd47ca754c5a`
- Model: `meta-llama/Llama-2-13b-hf`
- GPU: NVIDIA RTX 6000 Ada Generation, compute capability 8.9
- Runtime: PyTorch `2.2.1+cu121`, CUDA runtime `12.1`
- Export: 280 packed linear tensors in two safetensors shards, 6.6 GB total
- Indexed checkpoint payload: 643 tensors; 280 `uint8` and 363 `float16`
- Export elapsed time: 33 minutes 7 seconds
- Peak export CPU RSS: 53,811,700 KiB (about 51.3 GiB)

The upstream source required recorded compatibility patches for the Ada build,
Transformers 4.38 cache/RoPE APIs, GPTQ V-projection naming and mask handling,
export device placement, and low-memory FP16 loading. The patches are retained
under `patches/`; they adapt build and framework interfaces rather than replace
the upstream quantization, GEMM, or KV-cache algorithms.

## Full-checkpoint smoke

`scripts/run_upstream_quarot_checkpoint_smoke.py` loaded the two checkpoint
shards and checked every packed decoder linear before two identical fixed-token
generation runs (`prefill=16`, `decode=2`). The following gates passed:

- exactly 280 upstream `Linear4bit` modules;
- every packed weight is `uint8`;
- every weight scale is finite and positive;
- prefill and decode logits are finite;
- final cache length is 18 in both runs;
- greedy token IDs are exactly repeated (`[29949, 332]`).

Peak allocated/reserved GPU memory was 7,188,692,480 / 7,451,181,056 bytes.
This complements the separately passed exact integer-accumulator and KV-cache
primitive checks. It is a load/execution/determinism gate, not a PPL result or
a claim of logit equivalence to FP16.

## Matched performance result

`scripts/run_upstream_quarot_benchmark.py` loads FP16 and W4A4KV4 sequentially,
uses identical deterministic token IDs, synchronizes CUDA around each timed
region, excludes loading/conversion from timing, performs warm-up, and retains
all raw samples. The table reports medians.

| Workload | Metric | Upstream FP16 | Official W4A4KV4 | FP16 / W4 ratio | Interpretation |
|---|---:|---:|---:|---:|---|
| 128 prefill + 8 decode; 1 warm-up, 3 repeats | prefill | 53.38 ms | 71.14 ms | 0.750x | W4 is 1.33x slower |
| same | decode | 53.67 ms/token | 77.73 ms/token | 0.690x | W4 is 1.45x slower |
| same | end to end | 485.78 ms | 698.53 ms | 0.695x | W4 is 1.44x slower |
| 2048 prefill + 32 decode; 3 warm-ups, 10 repeats | prefill | 392.51 ms | 441.70 ms | 0.889x | W4 is 1.13x slower |
| same | decode | 52.28 ms/token | 79.51 ms/token | 0.658x | W4 is 1.52x slower |
| same | end to end | 2076.90 ms | 2945.97 ms | 0.705x | W4 is 1.42x slower |

For the 2048+32 workload, allocated memory was:

| Measurement | Upstream FP16 | Official W4A4KV4 | W4 / FP16 |
|---|---:|---:|---:|
| model resident | 26.292 GB | 7.179 GB | 27.30% |
| prefill peak | 30.221 GB | 8.635 GB | 28.57% |
| decode peak | 28.490 GB | 8.183 GB | 28.72% |
| end-to-end peak | 28.796 GB | 8.395 GB | 29.15% |

All timed FP16/W4 paths produced finite logits, and all 280 packed modules were
present during the W4 measurements. The negative latency result should be
presented directly: this legacy upstream kernel substantially reduces memory
on Ada but does not outperform the highly optimized FP16 baseline at batch 1.

## Why lower precision did not accelerate this workload

The measured result is consistent with the implementation rather than a
failure to enter the low-bit path:

1. The upstream CUTLASS GEMM selects `cutlass::arch::Sm80` in `gemm.cu`.
   Adding `sm_89` to the build makes the legacy backend executable on Ada, but
   does not supply an Ada-tuned kernel, shape autotuning, or a modern fused
   W4 implementation such as Marlin.
2. Each packed linear remains a multi-kernel pipeline. `Quantizer` first runs
   an absolute-value maximum reduction and scale calculation, then launches
   FP16-to-int4 packing. The GEMM writes a full int32 output, and a separate
   kernel reads that output to dequantize it to FP16. These extra launches and
   intermediate-memory operations can exceed the saved weight bandwidth.
3. Batch-one autoregressive decode gives each projection an effective GEMM
   height of one. Tensor Core occupancy is low while Llama-2-13B still invokes
   seven packed projections in each of 40 layers for every token, so many small
   kernel launches dominate decode latency.
4. QuaRot also retains online Hadamard transforms around the output/down
   projections and in the quantized cache path. They are required by this
   faithful algorithm but are not fused into the surrounding kernels.
5. KV4 reduces cache storage and attention bandwidth, but each update must
   compute scale/zero-point metadata, pack K/V, update page metadata, and
   dequantize inside the decode attention kernel. At a 2048-token context this
   saving did not amortize those costs across the whole decoder.
6. The comparator is already strong: it uses the same upstream model/cache
   structure with FP16 weights plus FlashAttention 2 and mature FP16 GEMM on an
   Ada GPU. Lower storage precision alone cannot guarantee lower latency.

The workload trend supports this explanation. Increasing prefill from 128 to
2048 tokens narrowed the W4 slowdown from 1.33x to 1.13x, consistent with
better amortization at larger matrix sizes. Decode remained 1.52x slower,
where the one-token matrix shape and launch overhead are least favorable.

No claim is made that W4A4 cannot accelerate on other kernels or workload
shapes. The supported conclusion is narrower: this pinned official backend,
on RTX 6000 Ada at batch one, provides a large memory-capacity improvement but
not latency acceleration.

## Recovered artifacts

The raw results are retained locally under the ignored results tree at
`results/quarot-real-deployment/runpod-rtx6000ada/full-llama2-13b-gptq-w4a4/`.

| Artifact | SHA-256 |
|---|---|
| `checkpoint-smoke.json` | `d721a13e03594b52b8dad9f7daf31be56ac8f0ceca3a0a701fc06ece5d8b61f0` |
| `checkpoint-smoke.log` | `48e5c7d1c4cac014c714ded09b4ea650d3522898ebcdf5012bdbb37edab92d81` |
| `benchmark-smoke.json` | `629d636a3c749a6f0e29903f87b526d86de49c23cd4d17cb7ad65a6a4538b7d1` |
| `benchmark-smoke.log` | `89b13dd175584cb875ee87f5a74f9d1456121269b494be812d3901f2397099f2` |
| `benchmark-2048-32.json` | `accfb85063a7c9382b33d0503d23daff190a733e04b64cf9b311d37ed6e839e8` |
| `benchmark-2048-32.log` | `89798d6d5ab0e512412155099f4b6119ba2176a54eec85feea8f05519309c680` |

The checkpoint itself remains on the RunPod persistent volume. The next Route
A accuracy gate is a bounded text/PPL slice if presentation-quality accuracy
evidence is required. Route B remains separate and should compare original
BF16, unrotated W4A16, and offline-rotated W4A16 under vLLM.
