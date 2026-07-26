# Official QuaRot single-block RunPod runbook

> Completed on RTX 6000 Ada on 2026-07-22. The accepted measurements and the
> retained FP16 batch-64 capacity OOM are recorded in
> `OFFICIAL_QUAROT_SINGLE_BLOCK_RESULTS.md`.

## Purpose and evidence boundary

This run reproduces the performance scope used by the QuaRot paper: one
Llama-2 decoder block, not a complete language model. It uses the pinned
upstream FP16 and W4A4KV4 layer/cache implementations and records prefill,
decode, layer-e2e, and peak allocated-memory samples.

The target is one NVIDIA RTX 6000 Ada. The paper used an RTX 3090, so this run
is **paper-protocol-aligned**, not a hardware-exact numerical reproduction.
There is no pass/fail speedup threshold: a completed negative result remains a
valid result. This benchmark is also separate from the completed full-model
Llama-2-13B measurement and from the later QuaRot-style vLLM W4A16 route.

The upstream `e2e/benchmark_layer.py` is the behavioural reference. The project
runner keeps its layer classes, cache implementations, and timing structure:
three warm-up calls, ten timed calls per repetition, and ten repetitions. The
project runner additionally fixes the random seed, supplies the position IDs
required by Transformers 4.38, preserves every timing/memory sample, and writes
structured JSON.

## Frozen inputs

| Item | Value |
|---|---|
| Upstream QuaRot | `5008669b08c1f11f9b64d52d16fddd47ca754c5a` |
| Model | `meta-llama/Llama-2-7b-hf` |
| Model revision | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| GPU | One NVIDIA RTX 6000 Ada, compute capability 8.9 |
| Host toolkit | CUDA 12.8 (`nvcc 12.8.93`) |
| PyTorch | `2.2.1+cu121` |
| Transformers | `4.38.0` |
| FlashAttention | `2.5.6` |
| Plan | `configs/deployment/quarot_official_single_block_rtx6000ada.json` |

Only the Ada build and cache/RoPE Transformers-4.38 compatibility patches are
permitted. The `sm_89` build patch is required because the pinned upstream
build targets stop at `sm_86`. GPTQ/export patches are unrelated because the
upstream layer performance path constructs a shape-correct W4A4 layer rather
than loading a quantized quality checkpoint.

## Local gate already available

From the project root:

```bash
python3 scripts/run_upstream_quarot_layer_benchmark.py --validate-only
python3 -m unittest tests.test_upstream_quarot_layer_benchmark
```

These commands validate only the plan and reporting logic. macOS does not
produce CUDA performance evidence.

## 1. Start the RTX 6000 Ada Pod and record the untouched host

Use one RTX 6000 Ada with a persistent volume mounted at `/workspace`, at least
50 GB container storage, and at least 80 GB free persistent storage. The
accepted session uses Ubuntu 24.04, CUDA toolkit 12.8.93, and driver 570.195.03.

```bash
set -euo pipefail
cd /workspace/NewSmallProject
result_dir=results/quarot-official-single-block-rtx6000ada
mkdir -p "$result_dir"
nvidia-smi | tee "$result_dir/nvidia-smi-before.txt"
/usr/local/cuda/bin/nvcc --version | tee "$result_dir/nvcc-before.txt"
uname -a | tee "$result_dir/uname-before.txt"
```

Reject the Pod if the device is not an RTX 6000 Ada or the toolkit/runtime pair
does not match the frozen record. Do not silently switch GPU or runtime.

## 2. Prepare fresh source checkouts

Obtain the current project branch under `/workspace/NewSmallProject`, then use
a fresh upstream checkout. Do not reuse a patched Ada checkout.

```bash
cd /workspace/NewSmallProject
git status --short --branch

git clone --recurse-submodules https://github.com/spcl/QuaRot.git QuaRot-single-block
git -C QuaRot-single-block checkout 5008669b08c1f11f9b64d52d16fddd47ca754c5a
git -C QuaRot-single-block submodule update --init --recursive

git -C QuaRot-single-block apply --check ../patches/quarot-sm89-build.patch
git -C QuaRot-single-block apply ../patches/quarot-sm89-build.patch
git -C QuaRot-single-block apply --check ../patches/quarot-transformers-4.38-cache.patch
git -C QuaRot-single-block apply ../patches/quarot-transformers-4.38-cache.patch
git -C QuaRot-single-block apply --check ../patches/quarot-transformers-4.38-rope.patch
git -C QuaRot-single-block apply ../patches/quarot-transformers-4.38-rope.patch

git -C QuaRot-single-block diff --check
git -C QuaRot-single-block diff --name-only
git -C QuaRot-single-block submodule status
```

The only tracked upstream changes must be:

```text
e2e/quantized_llama/modeling_llama.py
quarot/transformers/kv_cache.py
setup.py
```

## 3. Reuse and verify the pinned official environment

```bash
source /workspace/NewSmallProject/.venv-quarot-official/bin/activate
python -c 'import torch, transformers, flash_attn; print(torch.__version__, torch.version.cuda, transformers.__version__, flash_attn.__version__)'

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export QUAROT_SOURCE=/workspace/NewSmallProject/QuaRot-single-block
export MAX_JOBS=8

mkdir -p results/quarot-official-single-block-rtx6000ada
python -m pip install -e ./QuaRot-single-block --no-build-isolation \
  2>&1 | tee results/quarot-official-single-block-rtx6000ada/build.log
python scripts/runpod_preflight.py \
  --output results/quarot-official-single-block-rtx6000ada/preflight.json
python -m pip freeze > results/quarot-official-single-block-rtx6000ada/pip-freeze.txt
```

The build must compile the upstream CUDA extension for `sm_89`. A successful
Python installation without a loadable `quarot._CUDA` extension does not pass.

## 4. Cache the pinned gated model

Set one persistent Hugging Face cache. The benchmark environment retains the
old Hub client required by Transformers 4.38, so use a separate authentication
environment for the browser device-code flow. Never write a token into a
command or log. Download only the safetensors format; downloading the complete
repository also retrieves a second 13.5 GB `.bin` copy.

```bash
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE=/workspace/hf-cache
python3 -m venv /workspace/hf-auth-venv
/workspace/hf-auth-venv/bin/python -m pip install 'huggingface_hub==1.24.0'
/workspace/hf-auth-venv/bin/hf auth login --format agent
export HF_HUB_DOWNLOAD_TIMEOUT=120
/workspace/hf-auth-venv/bin/hf download meta-llama/Llama-2-7b-hf \
  config.json generation_config.json model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json \
  --revision 01c7f73d771dfac7d292323805ebc428287df4f9 \
  --cache-dir "$HF_HUB_CACHE" --max-workers 1
```

## 5. Recheck the official kernels and tiny e2e path

Run the official integer, KV-cache, and tiny model gates in the fresh SM89
checkout before timing the 7B layer:

```bash
python scripts/run_upstream_quarot_primitive_smoke.py \
  --output results/quarot-official-single-block-rtx6000ada/primitive.json
python scripts/run_upstream_quarot_kv_smoke.py \
  --output results/quarot-official-single-block-rtx6000ada/kv.json
python scripts/run_upstream_quarot_tiny_e2e_smoke.py \
  --output results/quarot-official-single-block-rtx6000ada/tiny-e2e.json
```

All three commands must exit zero. These checks establish that the official
CUDA components execute correctly on the new Pod; they are not performance
results.

## 6. Run the small layer-performance gate

```bash
set -o pipefail
python scripts/run_upstream_quarot_layer_benchmark.py \
  --source QuaRot-single-block \
  --groups smoke \
  --local-files-only \
  --output results/quarot-official-single-block-rtx6000ada/smoke.json \
  2>&1 | tee results/quarot-official-single-block-rtx6000ada/smoke.log
```

Do not start the formal matrix unless the command exits zero and reports
`OFFICIAL_QUAROT_SINGLE_BLOCK_STATUS=passed`.

## 7. Run the paper-aligned matrix

```bash
python scripts/run_upstream_quarot_layer_benchmark.py \
  --source QuaRot-single-block \
  --groups paper_prefill paper_decode_memory \
  --local-files-only \
  --output results/quarot-official-single-block-rtx6000ada/formal.json \
  2>&1 | tee results/quarot-official-single-block-rtx6000ada/formal.log
```

The runner returns nonzero when the unchanged formal grid completes with one or
more structured OOM points. In that case, inspect `formal.json.status` and the
per-case `oom_by_mode` records; do not discard the completed measurements.

The prefill group uses sequence length 2048 and batch sizes `1, 4, 16, 64`.
The decode/memory group uses batch sizes `1, 16`, context lengths
`256, 512, 1024, 2048, 4096`, and 50 decode tokens. The latter follows Figure
4's explicit “decoding of 50 tokens” wording; Table 17's “single token” caption
is retained as a paper ambiguity and is not silently merged into this result.

## Acceptance checklist

- `formal.json.status` is `"passed"` when every point fits, or
  `"completed_with_oom"` when an unchanged paper-grid point exceeds GPU
  capacity; every OOM must retain its mode, metric, and allocator message;
- runtime is exactly RTX 6000 Ada / `sm_89` / Torch `2.2.1+cu121` / Transformers
  `4.38.0` / FlashAttention `2.5.6`;
- upstream revision and the three allowed tracked compatibility paths match;
- primitive, KV-cache, and tiny-e2e official-backend gates pass before timing;
- every W4A4KV4 layer contains seven upstream `Linear4bit` projections and
  every FP16 layer contains zero;
- every successfully executed formal metric retains ten positive finite timing
  samples and ten positive peak-memory samples;
- FP16/W4 speed and memory ratios exist wherever both modes fit; OOM points are
  reported as capacity limits and are never replaced with a smaller workload;
- no numerical speedup threshold is imposed.

This acceptance proves only an official-backend, single-decoder-block RTX 6000 Ada
performance result. It does not prove full-model latency, model quality, an RTX
3090 reproduction, or the later vLLM W4A16 serving path.
