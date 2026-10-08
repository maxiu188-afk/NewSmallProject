# Low-Bit LLM Quantization and Deployment

This repository is a reproduction and systems study of rotation-based low-bit
LLM inference. It follows QuaRot and SpinQuant from algorithmic fake
quantization through packed checkpoints, custom CUDA experiments, and real
serving with vLLM and SGLang.

The project uses Llama-2 as its main full-scale test case and keeps accuracy,
memory, kernel performance, and serving performance as separate measurements.

## What I did

- Reproduced QuaRot and SpinQuant transformations, including residual, V/O,
  Q/K-after-RoPE, and structured MLP Hadamard rotations.
- Built a configuration-driven PyTorch pipeline for W4, W4A4, and W4A4KV4
  experiments, with matched naive and rotated controls.
- Implemented and validated an owned packed-W4/A8 integer-accumulator path for
  the three Llama-2-13B Linear shape classes and all 280 decoder projections.
- Exported rotation-aware GPTQ checkpoints and deployed W4A16 and W4AFP8
  variants through vLLM on NVIDIA GH200.
- Evaluated model quality with WikiText-2 and BoolQ, and measured serving
  throughput, TTFT, end-to-end latency, and GPU memory.
- Compared vLLM and SGLang under a matched serving protocol, including a
  controlled SGLang overlap-scheduling ablation.

## Selected results

| Study | Main result |
|---|---|
| QuaRot GPTQ quality | Llama-2-13B WikiText-2 PPL **5.8376**, versus **5.0087** BF16 and **8624.35** for the matched naive W4A4 control |
| Packed W4A4KV4 memory | Model-resident allocation reduced from **26.29 GB to 7.18 GB** on RTX 6000 Ada |
| vLLM W4A16 serving | **1.37--1.54x** request throughput and **52.7%** lower ready GPU memory than BF16 on GH200 |
| Rotated W4A16 quality | WikiText-2 PPL **5.1328** versus **5.0078** BF16; BoolQ **80.76%** versus **80.58%** BF16 |
| SpinQuant W4A16 | **1.38--1.54x** request throughput with about **52%** lower ready GPU memory |
| Backend comparison | SGLang delivered **20.45%** higher median throughput at concurrency 1 and **1.07%** at concurrency 8; vLLM retained lower TTFT |

These results show the full path from a rotation-based quantization idea to a
real packed checkpoint and an OpenAI-compatible serving backend. They also
show why a method should be judged on both model quality and deployment
behaviour: compression alone does not determine end-to-end speed.

## Project progression

```text
rotation primitives and fake quantization
        -> full-model GPTQ quality
        -> packed W4/A8 CUDA path
        -> official QuaRot W4A4KV4 backend
        -> vLLM W4A16 / W4AFP8 serving
        -> SpinQuant transfer and vLLM/SGLang comparison
```

## Repository structure

```text
repro/          Portable QuaRot and SpinQuant implementations
configs/        Model, quantization, calibration, and deployment configs
scripts/        Local, RunPod, and Isambard experiment entry points
csrc/           Owned packed low-bit CUDA experiments
tests/          Algebra, pipeline, checkpoint, and deployment checks
docs/           Result summaries, protocols, runbooks, and archive index
results/        Tracked local evidence and compact result records
environments/   Reproducible environment definitions
patches/        Audited compatibility patches for external backends
QuaRot/         Local pinned upstream QuaRot reference checkout
SpinQuant/      Local pinned upstream SpinQuant reference checkout
```

Large checkpoints and server-only artifacts are intentionally kept outside
Git. The repository stores configs, manifests, compact results, and the code
needed to reconstruct each experiment.

## Project status

The planned accuracy, packed-kernel, vLLM, W4AFP8, SpinQuant, and
SGLang-versus-vLLM studies are complete. The repository is retained as a
reproducible research record and as a base for future low-bit deployment work.

## Start here

1. [Phase status](docs/PHASE_STATUS.md) -- concise project state and headline
   results.
2. [Documentation index](docs/README.md) -- all current result documents and
   runbooks, grouped by topic.
3. [Fake-quant results](docs/FAKE_QUANT_RESULTS.md) -- algorithmic QuaRot
   reproduction and matched quality controls.
4. [Official QuaRot results](docs/OFFICIAL_QUAROT_RESULTS.md) -- packed
   W4A4KV4 full-model and block-level evidence.
5. [vLLM W4A16 results](docs/VLLM_W4A16_RESULTS.md) -- deployed checkpoint
   quality, memory, throughput, and BoolQ.
6. [W4AFP8 results](docs/W4AFP8_RESULTS.md) and
   [serving-backend comparison](docs/SERVING_BACKEND_COMPARISON_RESULTS.md) --
   later deployment studies.
7. [Portable pipeline guide](docs/PORTABLE_PIPELINE.md) -- local setup and
   runnable entry points.

The upstream QuaRot reproduction plan is retained in
[QUAROT_REPRODUCTION_PLAN.md](QUAROT_REPRODUCTION_PLAN.md), and completed
historical runbooks are indexed under [docs/archive](docs/archive/README.md).
