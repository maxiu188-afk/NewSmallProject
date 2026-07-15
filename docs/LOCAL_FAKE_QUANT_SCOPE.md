# Local fake-quant smoke scope

The local fake-quant ablation uses a deterministic random two-layer LLaMA only.
It applies symmetric QDQ to every linear weight per output channel and, when
enabled, to every linear input per token.

It compares five controlled cases: FP32, naive W4, rotated W4, naive W4A4, and
rotated W4A4. The output is logit deviation from the FP32 random-model result.

This test verifies that the quantization locations execute and that the naive
and rotated branches are separately observable. It cannot demonstrate QuaRot's
accuracy advantage: there are no pretrained weights, calibration data, language
model perplexity, GPTQ, K/V-cache quantization, low-bit kernel, memory, or
throughput measurements. Do not cite its numerical errors as paper reproduction
results.

Measured local output is recorded separately in `LOCAL_FAKE_QUANT_RESULTS.md`.
