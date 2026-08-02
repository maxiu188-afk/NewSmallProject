# W4AFP8 deployed-quality results

## Accepted result

Formal Isambard job `5874807` completed `0:0` on 2026-08-02 after passing
smoke job `5874806` through `afterok`. It evaluated the BF16 reference and all
three packed W4AFP8 checkpoints through vLLM over 162 non-overlapping
2048-token WikiText-2 sequences, scoring 331,614 next-token targets per model.

| Model | Total NLL | Mean NLL | PPL | PPL vs BF16 | PPL vs unrotated |
|---|---:|---:|---:|---:|---:|
| BF16 | 534,230.409917 | 1.611000772 | 5.007820 | baseline | -- |
| Unrotated W4AFP8 | 542,618.332650 | 1.636295008 | 5.136105 | +0.128285 (+2.5617%) | baseline |
| QuaRot-style W4AFP8 | 549,787.787161 | 1.657914886 | 5.248356 | +0.240536 (+4.8032%) | +0.112251 (+2.1855%) |
| SpinQuant-transfer W4AFP8 | 548,635.773268 | 1.654440926 | 5.230155 | +0.222335 (+4.4397%) | +0.094050 (+1.8312%) |

SpinQuant-transfer is 0.018201 PPL, or 0.3468%, lower than QuaRot-style, but
both rotated variants are worse than the matched unrotated W4AFP8 control.
Under this backend-compatible group-128/no-actorder/min-max recipe, neither
offline rotation improves deployed PPL. This finding is specific to the
declared recipe and checkpoint set; it is not a paper-protocol comparison.

## Provenance and acceptance checks

- Source revision: `1f3e4cb2afc81b381b474cd5ebe3c32aa55460bb`.
- Joint four-model source gate: job `5874345`, result SHA-256
  `6a22855a720b95e7230dd15644ced73b28297a960e29d5d2823666de28254f2a`.
- Backend capability SHA-256:
  `048fce61ad390e0ac4dfb89db2f1801c5f47468e993a5ffbdb6b4b4e7d61633a`.
- Retained-token manifest SHA-256:
  `b6ed5f122ba85a9752b3dda699d73405e5a86b7db19050c1f395cbbea450f1cf`.
- Token-ID SHA-256:
  `0f49a76a5cc6f3841356f09fee93eb5a8de9cc37d6b65af54e40614d6eac0de9`.
- Formal result SHA-256:
  `78a81dcdb17d8393247abd65d2f7e00b030e78817dc6d8b0699a8c15e48481f3`.
- The formal JSON identifies checkpoint paths but does not embed their content
  hashes. A post-run audit therefore hashed every file after completion; all
  recorded checkpoint mtimes predate the formal job. Tree SHA-256 values are
  `c38e6f693a466d02e4cdf032250facf22c3f931b86bf6e1663eb89c0b8592496`
  unrotated, `06792e63521d2fa74f0bfa061def342a379453045d6fbc1c034c18446d9a8bf3`
  QuaRot-style, and
  `2037afc16341dcf1d779184fabbfbef0152990ba4132274f717045dce47d8101`
  SpinQuant-transfer. This is a supplemental post-run provenance record, not a
  field produced inside job `5874807`. The supplemental manifest SHA-256 is
  `ce726d8dcc6100de89ee5df679b8b14f3f9b36515ff6d8f7d28714c5785ca8f9`.
- Every W4AFP8 checkpoint contains 280 packed decoder linears, no runtime
  `g_idx`, and compressed-tensors metadata with group size 128,
  `actorder=None`, and `memoryless_minmax` weights.
- The GH200 audit selected `CutlassW4A8LinearKernel` for every required Llama
  shape. The repeated DeepGEMM import warning was an unused optional-backend
  probe; all selected CUTLASS executions and result assertions passed.

Raw smoke/formal JSON and `checkpoint-sha256-postrun.json` remain in the
producing Isambard checkout under `results/vllm-w4afp8-llama2-13b/`, with local
ignored copies retained for review. They are intentionally excluded from Git
by the repository's generated-result policy; the hashes above bind this
reviewed summary to those artifacts.

## Evidence boundary

This closes the deployed-checkpoint PPL half of the W4AFP8 experiment. It does
not establish acceleration. Matched full-model serving smoke and formal
throughput/latency/memory measurement remain required before any W4AFP8
variant is labelled deployment-complete.

The SpinQuant checkpoint remains an INT8-trained rotation transfer to W4AFP8,
not an FP8-targeted learned endpoint. Its slightly better PPL than QuaRot does
not show that FP8-targeted SpinQuant training is complete.
