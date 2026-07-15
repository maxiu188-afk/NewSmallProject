# QuaRot method and execution note

This note documents the observed QuaRot reference behaviour and the
reproduction interpretation. It does not describe SpinQuant.

## Objective

Outliers in a Transformer residual stream make low-bit per-token quantization
costly. QuaRot applies orthogonal changes of basis that leave the
full-precision model function unchanged, then quantizes the more even
coordinate distributions.

Let the residual state be a row vector `x`, and let `Q` be orthogonal. QuaRot
represents that state as `x' = xQ`. For a linear operation written as
`F(x, W) = x W^T`:

| Location | Transformed weight / activation | Full-precision effect |
|---|---|---|
| residual input to Q/K/V or MLP up/gate | `W' = WQ` | `F(xQ, WQ) = F(x, W)` |
| residual-producing output (attention O or MLP down) | `W' = Q^T W` | output becomes `F(a, W)Q` in the new residual coordinates |
| embedding and output head | apply the matching residual basis change | token logits are unchanged |

LayerNorm/RMSNorm affine weights are fused into adjacent linear weights before
replacing them with weight-free normalization. This is why a rotation can pass
through the normalized residual stream without altering its norm.

## Online rotations that matter for quantization

Besides the residual rotation, the reference introduces Hadamard transforms at
specific non-residual locations:

| Path | Transformation purpose | Quantization consequence |
|---|---|---|
| MLP intermediate -> down projection | apply a full Hadamard online and absorb its inverse into down-proj weights | reduces down-proj input outliers before activation quantization |
| V -> O attention path | rotate V output and compensate in O input | makes V-cache / O-path values easier to quantize |
| Q/K after RoPE, per head | apply the same head-dimension Hadamard to Q and K | preserves attention dot products because `(qH)(kH)^T = qk^T` while helping K quantization |

For LLaMA-2 7B, the hidden size is 4096 and attention head dimension is 128,
both directly compatible with a power-of-two Hadamard. The MLP intermediate
size 11008 is factored as `172 * 64`; the 64-sized fast transform needs an
additional remainder matrix. A replacement implementation must preserve this
factorization rather than silently use a different transform.

## What the upstream fake-quant path actually measures

Observed in `QuaRot/fake_quant/`:

1. It fuses layer-norm weights and rotates model weights.
2. It applies weight quantization through GPTQ or RTN, but the evaluation path
   remains ordinary floating-point PyTorch linear execution using modified/
   dequantized weights.
3. `ActQuantWrapper` quantizes then dequantizes linear inputs per token or
   per token-group before calling the original linear layer.
4. The V-projection output can be QDQ-ed for V-cache simulation. K is rotated
   after RoPE and QDQ-ed in the corresponding wrapper.

Therefore F0–F5 measure algorithmic QDQ accuracy, not int4 kernel throughput,
packed checkpoint size, or end-to-end serving memory.

## What the upstream deployment path attempts to measure

Observed in `QuaRot/quarot/` and `QuaRot/e2e/`:

| Component | Intended low-bit mechanism | Status before RunPod validation |
|---|---|---|
| W4 linear | signed int4 packed two values per byte, custom CUDA int4 GEMM, then scaled dequantization | static code audit only |
| A4 input | per-token symmetric quantization before `Linear4bit` | static code audit only |
| KV4 cache | asymmetric int4 packing plus per-head scale/zero and paged CUDA attention | static code audit only |
| E2E LLaMA | converted checkpoint, custom modules, FlashAttention-2-oriented path | static code audit only |

The code makes low-bit storage and custom kernels plausible, but no deployment
claim is valid until a RunPod build and numerical comparison demonstrate that
the selected path actually executes on the target GPU.

## Local evidence and its limit

The local pure-Python toy test checks the equations jointly: RMSNorm fusion,
residual rotation, Q/K dot-product preservation, V/O compensation, MLP online
Hadamard compensation, and output-head logits. The additional random tiny
PyTorch LLaMA smoke test checks the actual Transformers module graph for
RMSNorm fusion, residual rotation, the static V/O compensation pair, MLP online
Hadamard compensation, and output-head logits. Q/K after-RoPE injection remains
covered only by the framework-free test because it is an attention-internal
monkeypatch in the upstream code.

It has one-token attention, no PyTorch module graph, no pretrained weights, no
KV allocation, and no low-bit CUDA operation. It establishes a necessary
mathematical gate, not a model-level or deployment-level reproduction.
