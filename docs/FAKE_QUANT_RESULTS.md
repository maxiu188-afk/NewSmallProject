# Consolidated fake-quant results

## Scope and evidence boundary

This is the current summary of the repository's algorithmic fake-quant work.
It replaces separate top-level notes for the local smokes, SmolLM2 checks,
Llama-2-13B BF16 control, RTN pair, GPTQ result, and calibration/component
study. Their detailed records remain under [`archive/fake_quant/`](archive/fake_quant/).

All formal low-bit rows below are floating-point QDQ simulations. They do not
demonstrate packed W4 or KV storage, integer kernels, latency, throughput, or
memory reduction. The formal F3/F4 rows use W4A4 with K/V left at 16 bits.

## Formal Llama-2-13B protocol

| Item | Frozen value |
|---|---|
| Model | `meta-llama/Llama-2-13b-hf` at `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` |
| Evaluation data | `Salesforce/wikitext`, `wikitext-2-raw-v1`, test revision `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Evaluation | 162 non-overlapping 2048-token sequences; 331,614 next-token targets; seed 0 |
| GPTQ calibration | Pinned train split; 128 complete 2048-token sequences for the primary rows |
| GPTQ configuration | Symmetric group/block-128 W4; 1% Hessian damping; activation ordering enabled |
| Model coverage | 40 decoder layers and 280 linear layers |

## Primary accuracy result

| Row | Weight fitting | Rotation | W/A/K/V | PPL |
|---|---|---|---:|---:|
| F0 BF16 reference | none | none | 16/16/16/16 | 5.0087 |
| F3 naive RTN | RTN QDQ | none | 4/4/16/16 | 8719.6775 |
| F4 QuaRot RTN | RTN QDQ | complete | 4/4/16/16 | 12.4606 |
| F3 naive GPTQ | GPTQ | none | 4/4/16/16 | 8624.3509 |
| F4 QuaRot GPTQ | GPTQ | complete | 4/4/16/16 | **5.8376** |

The naive W4A4 path is catastrophic under both RTN and GPTQ. QuaRot reduces
the RTN PPL from 8719.68 to 12.46; GPTQ then reduces the complete QuaRot result
to 5.84, only 0.83 PPL above the same-run BF16 control. The supported conclusion
is that the rotations are essential for this 4-bit configuration and that GPTQ
substantially improves the already rotated model. GPTQ alone does not rescue
the naive path.

## Cumulative component study

| 128-sequence configuration | PPL |
|---|---:|
| Naive F3 | 8624.3509 |
| Residual Hadamard | 17.7284 |
| Residual + V/O | 30.7677 |
| Residual + V/O + MLP | 5.8452 |
| Complete F4 (+ post-RoPE Q/K) | **5.8376** |

This sequence is cumulative and uses one seed; it is not a factorial ablation.
The V/O step worsens the residual-only row before the MLP transform recovers
the model, so the individual changes must not be presented as isolated causal
effects. The final post-RoPE Q/K step improves the near-complete row by 0.0077
PPL.

## Calibration-size study

| Calibration sequences | Naive F3 PPL | Complete F4 PPL |
|---:|---:|---:|
| 32 | 7427.8833 | 5.8679 |
| 64 | 8231.3756 | 5.8478 |
| 128 | 8624.3509 | 5.8376 |

Complete F4 is stable across the tested budgets: 32 and 64 sequences add only
0.0303 and 0.0102 PPL relative to 128. The naive result remains catastrophic
and non-monotonic, so it does not support a claim that more calibration data
improves the naive implementation.

## Preliminary checks retained for provenance

Before the formal 13B study, the project passed deterministic local primitive,
tiny random-Llama, SmolLM2-135M, Windows CUDA, and RunPod A40 smokes. The A40
matched synthetic-token W4A4KV4 check produced mean/max logit errors of
8.3990/44.7423 for naive and 5.0678/33.9175 for QuaRot. These checks established
that the intended code paths executed and that the branches were distinct;
they are not quality or performance results.

## Detailed archived records

- [`LLAMA2_13B_BF16_BASELINE.md`](archive/fake_quant/LLAMA2_13B_BF16_BASELINE.md)
- [`LLAMA2_13B_RTN_W4A4_RESULTS.md`](archive/fake_quant/LLAMA2_13B_RTN_W4A4_RESULTS.md)
- [`LLAMA2_13B_GPTQ_W4A4_RESULTS.md`](archive/fake_quant/LLAMA2_13B_GPTQ_W4A4_RESULTS.md)
- [`LLAMA2_13B_GPTQ_ABLATION_CALIBRATION_RESULTS.md`](archive/fake_quant/LLAMA2_13B_GPTQ_ABLATION_CALIBRATION_RESULTS.md)
- [`RUNPOD_FAKE_QUANT_RESULTS.md`](archive/fake_quant/RUNPOD_FAKE_QUANT_RESULTS.md)
