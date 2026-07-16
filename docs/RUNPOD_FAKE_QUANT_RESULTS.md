# RunPod CUDA fake-quant record

## Scope

This is a small server-side numerical smoke, not a perplexity, zero-shot, or
throughput result. It uses one deterministic synthetic token batch to confirm
that the portable QDQ path executes on CUDA and that the matching naive and
QuaRot variants can be compared on the same model and inputs.

Raw JSON, command logs, and per-run configurations are retained in the
ignored server directory `results/runpod-fake-quant/` on the persistent
`/workspace/NewSmallProject` volume. The Pod is stopped; the Python virtual
environment was deliberately on disposable container storage and must be
rebuilt for a later session.

## Recorded environment

| Field | Recorded value |
|---|---|
| Date | 2026-07-16 |
| GPU | NVIDIA A40, 46,068 MiB, compute capability `8.6` |
| Driver | `570.211.01` |
| CUDA compiler | `12.8.93` |
| Python | `3.11.13` |
| PyTorch | `2.11.0+cu128` (`torch.version.cuda == 12.8`) |
| Transformers | `5.14.1` |
| Project revision | `e36fedf40e47085c606836eac49b843292a07949` |
| Upstream revision | `5008669b08c1f11f9b64d52d16fddd47ca754c5a` |

The preflight and post-install records are respectively
`results/runpod-preflight/preflight-initial.json` and
`results/runpod-preflight/preflight-cu128.json` on that volume.

## CUDA numerical gate

The 16-bit run used `HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, float32, CUDA, a fixed seed, and
one synthetic batch of 32 tokens. Residual, V/O, MLP-online, and Q/K-after-RoPE
rotation were enabled.

| Check | Mean absolute logit error | Maximum absolute logit error |
|---|---:|---:|
| QuaRot W16A16KV16 vs reference | `1.347205e-05` | `2.908707e-04` |

This passes the server numerical gate for this small fixed-input scope. It is
not a native CUDA-extension or packed-int4-kernel result.

## Matched W4A4KV4 smoke

The model revision, dtype, CUDA device, token batch, seed, and 4-bit QDQ widths
were fixed between rows. The sole experimental difference was whether the
QuaRot rotation features were enabled.

| Variant | Mean absolute logit error | Maximum absolute logit error |
|---|---:|---:|
| Naive W4A4KV4 | `8.3990068` | `44.7422943` |
| QuaRot W4A4KV4 | `5.0678024` | `33.9174576` |

The QuaRot row has lower error in this narrow smoke. Neither value is a text
perplexity or quality measurement, and the implementation remains floating
point QDQ simulation rather than packed low-bit execution.

## Reference-build status

The unchanged upstream `pip install -e .` attempt was recorded before the
smoke. Modern pip created an isolated build environment without PyTorch, while
the upstream setup scripts import PyTorch. An outer no-isolation retry reached
CMake configuration but the nested fast-Hadamard editable install again used
isolation and failed for the same reason. `wheel` and `ninja` were installed in
the disposable server environment for diagnosis; no source patch was made and
the native kernel track was stopped rather than treated as successful.

## Next formal text evaluation

The chosen official dataset is `wikitext2`, which is the upstream fake-quant
default for both evaluation and GPTQ calibration. A strict upstream run also
requires an accessible, pinned LLaMA-2 checkpoint; the existing SmolLM2 result
does not substitute for that model requirement. The next Pod session should
rebuild the container environment, keep model/data/output under `/workspace`,
record the WikiText-2 revision and LLaMA-2 revision, then run the same
configuration for baseline and fake-quant variants.
