# Phase-2 full-decoder GPTQ W4A8 correctness gate

## Purpose and boundary

This is the next gate after the selected-`q_proj` layer/logits result. It
streams all 280 formally GPTQ-quantized Llama-2-13B decoder linears into a
self-describing sharded checkpoint, installs the exact packed values first in
an independent Torch integer oracle and then in the owned CUDA W4A8 modules,
and compares every decoder-layer output plus final logits for fixed tokens.

The decoder weights are packed signed W4 with FP32 group scales and their
per-linear GPTQ act-order input permutations. Runtime linear inputs use
per-token symmetric A8. Embeddings, `lm_head`, and K/V remain BF16. Passing
this gate does not establish PPL, KV4, latency, throughput, or memory savings.

## Local preparation completed

The local code path includes:

- streaming GPTQ capture, so completed packed tensors are moved to CPU and
  written immediately instead of accumulating all packed weights on the GPU;
- one `.pt` shard per decoder linear and an atomic `manifest.json` published
  only after every expected tensor is present;
- SHA-256, dtype, shape, permutation, bias, model, config, rotation, GPTQ, and
  source-revision, precision metadata validation;
- a full-checkpoint loader that rejects missing or extra model linears;
- fixed-token comparisons for all 40 layer outputs and final logits, followed
  by a short greedy-generation finite-output smoke with BF16 K/V.

Local macOS tests validate checkpoint streaming, checksum rejection, loading,
and reference-to-CUDA module conversion without executing the CUDA kernel.
Formal numerical evidence requires an NVIDIA CUDA allocation.

## Isambard execution

Isambard remains the primary experiment environment. Once the service is
healthy and the current selected-linear gate has been reviewed, synchronize a
clean committed revision to `$HOME/NewSmallProject`, then validate the batch
request before submission:

```bash
cd "$HOME/NewSmallProject"
sbatch --test-only scripts/run_isambard_w4a8_full_model_gate.sbatch
sbatch scripts/run_isambard_w4a8_full_model_gate.sbatch
```

The requested 24-hour wall time is intentionally not treated as validated
until `sbatch --test-only` accepts it on the restored service. Do not shorten
the numerical protocol merely to fit an unverified queue limit.

The job emits these stage markers:

```text
W4A8_FULL_GATE_STAGE=module_load
W4A8_FULL_GATE_STAGE=inputs
W4A8_FULL_GATE_STAGE=source_manifest
W4A8_FULL_GATE_STAGE=preflight
W4A8_FULL_GATE_STAGE=kernel_smoke
W4A8_FULL_GATE_STAGE=checkpoint
W4A8_FULL_GATE_STAGE=full_model_smoke
W4A8_FULL_GATE_STAGE=assert_result
```

If a complete checkpoint manifest already exists, the job validates and
reuses it. If tensor shards exist without a manifest, export stops rather than
silently mixing shards from different runs. Preserve the partial directory and
Slurm logs for diagnosis, then use a new empty output directory or remove only
the explicitly reviewed failed-run shards.

The formal batch job also rejects a dirty checkout and rejects reuse of a
checkpoint created by a different project revision or source configuration.

Success requires all of the following:

- `manifest.json` has exactly 280 indexed, checksum-valid tensor shards;
- all 40 captured layer outputs and the final logits pass the declared
  `atol=2e-4`, `rtol=1e-5` comparison;
- greedy-generation logits remain finite with BF16 K/V;
- the job prints `ISAMBARD_PHASE2_FULL_MODEL_GATE_PASSED`.

Only after these artifacts are recovered and reviewed may a matching W4A8 PPL
gate be prepared. KV4 and performance remain later, separate work packages.

## RunPod fallback

RunPod is a fallback when Isambard cannot provide a usable allocation. Attach
the existing persistent volume at `/workspace`, reuse the existing model cache,
and create the CUDA virtual environment on container disk. Run the same three
commands rather than a separate implementation:

```bash
python scripts/run_w4a8_cuda_smoke.py \
  --output results/phase2-w4a8-full-model/runpod-w4a8-kernel-smoke.json

python scripts/export_llama_gptq_w4a8_checkpoint.py \
  --config configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_gptq.json \
  --output-dir results/phase2-w4a8-full-checkpoint \
  --report results/phase2-w4a8-full-model/runpod-checkpoint-export.json

python scripts/run_llama_gptq_w4a8_full_model_smoke.py \
  --config configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_gptq.json \
  --manifest results/phase2-w4a8-full-checkpoint/manifest.json \
  --output results/phase2-w4a8-full-model/runpod-full-model-smoke.json \
  --sequence-length 16 --generation-tokens 8
```

Record the actual Python executable, environment, GPU, source revision, and
artifact hashes. A RunPod pass is independent evidence; it does not replace a
later Isambard portability result.
