# Qwen2.5-7B quantization research roadmap

## Decision

`Qwen/Qwen2.5-7B` is the next primary model. It is an openly accessible 7.61B
parameter decoder with Apache-2.0 licensing, RoPE, SwiGLU, RMSNorm, QKV bias,
and grouped-query attention (28 query heads and 4 KV heads). The target is
large enough to make low-bit behavior meaningful while avoiding the access
approval required by the original LLaMA-2 checkpoint.

`wikitext2` remains the common text corpus, matching the upstream QuaRot
fake-quant defaults. The exact Qwen model, tokenizer, and dataset revisions
must be resolved and written into each server-side manifest before results are
reported. No mutable model branch name is accepted as experimental provenance.

## Research position

QuaRot is a baseline, not the intended final system. Its reference checkout is
used to understand valid transformations and to reproduce a fixed/random
rotation control. New code remains in this project and is tested independently.
The first extension directions are:

1. Compare fixed random/Hadamard rotations with an optimized or learned
   rotation parameterization, inspired by SpinQuant.
2. Compare uniform W4A4KV4 simulation with a separately evaluated practical
   W4A8 path; QQQ is a useful reference for smoothing, compensation, and
   kernel-aware design.
3. Build owned packed-low-bit kernels only after their dequantized numerical
   reference passes. An upstream CUDA extension is not a prerequisite or an
   implementation target.

The project may use an open LLaMA-compatible model such as OpenLLaMA later as
a compatibility control, but Qwen2.5-7B is the primary research target.

## Ordered execution plan

| Gate | Work | Evidence required before proceeding |
|---|---|---|
| Q0 | Add a configuration-driven Qwen adapter; test random/synthetic Qwen-shaped models locally | Full-precision logits and selected hidden states agree with the unmodified reference within a declared tolerance |
| Q1 | Start a Pod, rebuild the disposable CUDA environment, and resolve Qwen2.5-7B plus WikiText-2 on `/workspace` | Saved immutable model/tokenizer/dataset revisions, environment snapshot, and full-precision WikiText-2 PPL |
| Q2 | Run matched no-rotation, QuaRot-style, and extension fake-quant rows | Same model/data/seed/context/calibration policy; raw PPL and numerical logs for every row |
| Q3 | Implement owned packed-W4 and then W4A8 execution checks | Output agreement against the dequantized reference; explicit precision labels for every tensor and GEMM |
| Q4 | Measure verified paths only | Separate prefill/decode timings, repeated synchronized samples, memory, and implementation revision |

## Boundaries

- The existing SmolLM2 and A40 synthetic-input runs remain useful smoke
  evidence only; they are not Qwen or WikiText-2 results.
- Qwen support must be derived from `model.config`; no model ID may be
  special-cased in the portable pipeline.
- QQQ, SpinQuant, and other recent work are literature and design references.
  A result may claim only the methods implemented and evaluated in this
  repository.
- Data, downloaded weights, CUDA environments, generated configurations, and
  raw results remain ignored artifacts on the persistent volume; the tracked
  repository holds source, reviewed configurations, and result summaries.

## References

- [Qwen2.5-7B model card](https://huggingface.co/Qwen/Qwen2.5-7B)
- [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)
- [QQQ: Quality Quattuor-Bit Quantization](https://arxiv.org/abs/2406.09904)
