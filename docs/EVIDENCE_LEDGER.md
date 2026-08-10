# Evidence ledger

This ledger maps each accepted claim to its primary reviewed record. It is a
documentation index, not a replacement for the raw result JSON, logs, source
manifests, or checkpoint manifests retained on the producing machine.

## Verification state

- **Accepted** means the result and provenance were reviewed and recorded in
  the linked current result document.
- **Locally indexed** means this checkout contains the reviewed summary and its
  recorded hashes.
- **Server recheck pending** means this documentation closeout has not yet
  re-read the producing-server files or recomputed their hashes. It does not
  invalidate an accepted result, but it remains the final durability audit.

As of 2026-08-09, all rows in the accepted evidence map below are accepted and
locally indexed. Jobs `5941763`, `5952554`, `5952594`, `5960180`, and `5961810`
and all of their bound artifacts were read directly from the producing server
and rehashed during acceptance. The historical-row producing-server
recheck remains pending and must be read-only: no training, export, inference,
evaluation, serving benchmark, or Slurm resubmission is authorized by this
ledger.

## Accepted evidence map

| Evidence track | Environment and primary gate | Accepted claim | Primary result binding | Current record |
|---|---|---|---|---|
| Algorithmic QuaRot fake quant | Llama-2-13B, WikiText-2, 162 x 2048 evaluation sequences | Complete QuaRot + GPTQ W4A4 floating QDQ reached PPL 5.8376 versus BF16 5.0087 | Frozen model/data revisions and protocol are recorded in the consolidated result; detailed run records remain archived | [`FAKE_QUANT_RESULTS.md`](FAKE_QUANT_RESULTS.md) |
| Official QuaRot backend | RTX 6000 Ada; upstream revision `5008669b08c1f11f9b64d52d16fddd47ca754c5a` | Full-model W4A4KV4 capacity result and paper-aligned single-block performance matrix | Exact artifact hashes and raw-sample summaries are retained in the two archived result records | [`OFFICIAL_QUAROT_RESULTS.md`](OFFICIAL_QUAROT_RESULTS.md) |
| Owned packed W4A8 correctness | RTX 6000 Ada full-decoder gate; GH200 selected-`q_proj` portability gate `5742443` | All 280 decoder linears matched the independent packed oracle for the fixed workload; no PPL or performance claim | Full-decoder manifest SHA-256 `caa12e485087e4bc5630c950e5b96041ba90e770cc40cdd7ae5defba64641e33`; result JSON SHA-256 `0a848e22ff1881ecd1ee76d829e3c39d43e2f9224892cdd1e61ada6be9b208d8` | [`W4A8_CUDA_KERNEL_RESULTS.md`](W4A8_CUDA_KERNEL_RESULTS.md) |
| QuaRot-style vLLM W4A16 | Isambard GH200; PPL `5839419`, serving `5780631`, layer diagnostic `5784967` | Packed-checkpoint PPL, matched full-model serving, and real layer-0 diagnostic are complete | PPL SHA-256 `f7698afcca494279cb7d3f2d50d94fd1378e03c6d6edd4506d750869cb09829a`; serving SHA-256 `2a683b38be4831f24897bb3d8660d3f0277c3a0c914aaebb5b8511f2f357a39a`; diagnostic SHA-256 `0a5e9658c3a2205b6e2465472f81bd004370ba7b4db1c46f07123fe3a538c5cb` | [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md) |
| SpinQuant-derived vLLM W4A16 | Isambard GH200; export/load `5945162`; BoolQ `5952594`; serving `5961810` | Learned offline R1/R2 was fused into a complete 280-linear packed W4A16 checkpoint; formal BoolQ reached 79.7554% versus 80.5810% BF16; matched serving reached 1.379--1.537x BF16 request throughput with 52.2--52.7% lower ready GPU memory | Export SHA-256 `b21ca19f34bf24470fdec797f90821b46edb77bfc1717c23724b989aa4ff05cf`; checkpoint-tree SHA-256 `41a79153d2ad7e0fb598819adc538ce65ba7c1a05629459f77cb816d226ce2da`; BoolQ SHA-256 `bdc6ac3263d14695321569b1c7b869fa11f24492e7ddbd82fb66315812549b64`; serving SHA-256 `8115da350e6c3c5c82f11a811f2e7ab6b1eda8bb8e504a3b3247a0f593e456e0` | [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md) |
| SGLang-vLLM QuaRot-style W4A16 serving | Isambard GH200; exact accepted checkpoint; BoolQ smoke `5941763`; serving smoke `5952554`; formal matrix `5960180` | Both backends loaded the same checkpoint; all 12 formal concurrency 1/8 x three-repetition cells passed. SGLang had 20.45% higher median request throughput at concurrency 1 and 1.07% at concurrency 8, but TTFT was 25.87% and 64.89% higher | BoolQ SHA-256 `62346ab360a025efc2636a9ab7a0d493ea7b2ae711c979b6c84139721d7aa5b0`; formal serving SHA-256 `fcaf3fef48dfa7d11f45fe20566789e95878938d106fdd539743425fb49d64be`; formal source-manifest SHA-256 `071c51aa46b2e290929ddf227521fa818aeb86dd70aee61f5307a938a45ed829` | [`SERVING_BACKEND_COMPARISON_RESULTS.md`](SERVING_BACKEND_COMPARISON_RESULTS.md) |
| SpinQuant fake-quant studies | Isambard GH200; W4A16 PPL `5847441`; corrected-A8 strong-GPTQ PPL `5854891` | Learned W4A16 R1/R2 ablation and corrected-A8 strong-GPTQ W4A8 diagnostic are complete | W4A16 PPL SHA-256 `e70ee79229dd3a19f8f1f7a588af3a32ebf76bdf745645d52019fd77941b1574`; corrected-A8 PPL SHA-256 `0b5403d3a0423efd9b3d7673107ed9b25d58350a7caf7643432aefcc2ef4c6fd` | [`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md) |
| W4AFP8 old INT8-trained transfer | Isambard GH200; PPL `5874807`, serving `5883004`, BoolQ `5876591` | Packed-checkpoint PPL, matched serving, and transfer-only BoolQ are complete | PPL SHA-256 `78a81dcdb17d8393247abd65d2f7e00b030e78817dc6d8b0699a8c15e48481f3`; serving SHA-256 `df63917675621f891280cf2cf5960e1b394a815f14fd8096125085ea02edae6f`; BoolQ SHA-256 `b278567aae262fdd6f4379d4004817f17faba51fa304077f03dd1479b8bdb824` | [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md) |
| W4AFP8 FP8-targeted SpinQuant | Isambard GH200; training `5876984`, source gate `5881273`, PPL `5884996`, serving `5884998`, BoolQ `5886914` | FP8-targeted rotation provenance, packed export/load, deployed PPL, matched serving, and BoolQ are complete | Training SHA-256 `f34f450e03dd59ec9b26942731b470b080908399f9d53089da2fd38bb903f1d1`; source-gate SHA-256 `c292e5ec7abfcfff762e5b1f59139fe0cb423b64cf24ba370969adf222e96a55`; PPL SHA-256 `9f17bc86ee664960400dd26f2182ceef72d76dce0a1b75f055331074a7ff7e0a`; serving SHA-256 `a10044d965d93ceca756e203a86ad5f72018a2b1028a65b552315a59ea1a80b7`; BoolQ SHA-256 `8e596efea2aeda06d705188060e06db081dd751f610bbef02002c972943f9a4a` | [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md) |

## Recent work status

No Slurm job is currently queued by these two workstreams. The formal
SGLang-vLLM concurrency 1/8 x three-paired-repetition matrix and the
SpinQuant-specific W4A16 serving benchmark are complete and accepted. New
experiments remain outside this ledger's authorization.

## Claim boundaries

- Fake-quant QDQ results are accuracy evidence only and do not establish
  packed storage, low-bit kernels, memory reduction, or acceleration.
- Official W4A4KV4, owned W4A8, vLLM W4A16, and vLLM W4AFP8 are separate
  runtime routes and must not share precision or performance labels.
- W4AFP8 acceleration is attributed to the deployed backend. The matched
  rotation variants differ by less than 1% in serving and do not establish a
  rotation-specific speedup.
- QuaRot-style W4A16 downstream-task quality, paper-GPTQ W4A8, online Hadamard
  R3/R4, and FP8 KV remain unmeasured. The separately scoped SpinQuant-derived
  W4A16 BoolQ result must not be generalized to those endpoints.
- The separately scoped SGLang-versus-vLLM study now accepts exact-checkpoint
  compatibility plus the formal concurrency 1/8 serving matrix. It does not
  establish BoolQ equivalence, generalize beyond the frozen W4A16 checkpoint,
  or prove one backend faster on every metric.
- SpinQuant W4A16 and FP8-targeted W4AFP8 now share frozen BoolQ and serving
  protocols, but their formal jobs were separate rather than interleaved; small
  cross-format performance differences remain descriptive.

## Pending read-only durability audit

When producing-server access is available, complete only these checks:

1. confirm the recorded raw JSON, logs, manifests, and checkpoint-hash reports
   still exist at the paths named by the current result documents;
2. recompute SHA-256 for the retained result and manifest files and compare
   them with this ledger and the primary records;
3. record any storage migration without changing the accepted result values;
4. do not rerun a missing artifact. If a file is absent, record the provenance
   gap explicitly and retain the reviewed summary as the available evidence.
