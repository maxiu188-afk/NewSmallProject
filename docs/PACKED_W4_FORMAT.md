# Owned packed-W4 format

This is the on-disk and numerical reference format for Phase 1.  It is owned
by this repository; the upstream QuaRot implementation was consulted only to
identify an int4 packing convention worth testing independently.

## Tensor contract

For a row-major linear weight `W[out_features, in_features]`:

- quantization is symmetric signed W4, groupwise along `in_features`;
- `group_size` divides `in_features` and is even;
- each group has one positive little-endian FP32 scale;
- quantized values are clamped to `[-8, 7]`;
- values `2j` and `2j+1` occupy byte `j`: the former is in the low nibble and
  the latter is in the high nibble, both in two's-complement representation;
- payload bytes are row-major, followed by no implicit padding.

`packed-w4-row-major-le-nibble-v1` is the format identifier.  A tensor is
stored as `<name>.w4.bin`, `<name>.scales.f32`, and
`<name>.packed-w4.json`.  The JSON manifest records the source provenance,
shape, group size, layout, and SHA-256 digests for both binary payloads.

## W4A8 numerical oracle

The Phase-1 reference quantizes an activation token symmetrically to A8 with
one positive scale.  It produces an int32 partial sum for each output row and
weight group, then applies the A8 scale and that group's W4 scale after the
integer dot product.  A CUDA kernel must first match these per-group int32
accumulators exactly; only then is its scaled FP16/BF16 result compared under a
predeclared floating-point tolerance.

## Initial supported shapes

The planned Llama-2-13B shapes are `(out_features, in_features)`:

- `(5120, 5120)` attention projections;
- `(13824, 5120)` MLP up/gate projections;
- `(5120, 13824)` MLP down projection.

The W4A8 kernel contract additionally requires `in_features % 32 == 0`.
Unsupported shapes must fail validation; they must not silently fall back to a
floating-point matmul while being labelled `int4_gemm`.

## Reference commands

The no-CUDA format smoke is safe to run locally:

```bash
python3 scripts/run_packed_w4_reference.py \
  --output results/local-packed-w4-reference/result.json
```

On the Phase-1 CUDA environment, a trusted saved 2-D GPTQ-transformed weight
tensor can be exported with explicit provenance:

```bash
python scripts/export_packed_w4_tensor.py \
  --input results/phase1-inputs/layer0-q-proj.pt \
  --tensor-name layer0_q_proj \
  --output-dir results/phase1-packed-w4 \
  --source-json results/phase1-inputs/layer0-q-proj-source.json
```

The exporter is an offline conversion tool.  Its output is `packed_weight`
evidence until the Phase-1 CUDA kernel matches the reference int32
accumulators.

Once a matching CUDA 12.8/PyTorch environment is recorded on RunPod, the
owned correctness kernel is run with:

```bash
python scripts/run_w4a8_cuda_smoke.py \
  --output results/phase1-w4a8-kernel/smoke.json
```

This is an integer-kernel correctness gate only.  It must report exact int32
accumulator agreement before the resulting path can be labelled `int4_gemm`;
it has no throughput or latency claim.
