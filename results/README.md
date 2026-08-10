# Generated result layout

`results/` contains generated artifacts and is ignored by default. Keep raw
model manifests, dataset cache references, logs, and JSON metrics on the
machine or persistent volume that produced them; commit only a reviewed summary
to `docs/` after the result boundary and provenance have been checked.

Historical RunPod fake-quant artifacts used this durable location:

```text
/workspace/NewSmallProject/results/llama2-13b-wikitext2-gptq-w4a4/
  quarot-f4-gptq.log       live tmux output
  quarot-f4-gptq.json      completed machine-readable result, if successful
```

The corresponding Hugging Face cache is
`/workspace/NewSmallProject/.cache/huggingface/`. It is intentionally not a
Git artifact. Before treating a server result as evidence, verify the exact
model/data revisions, evaluation token count, calibration split and token
count, CUDA/PyTorch runtime, process exit status, and finite metrics.

The accepted Isambard result families are retained under their producing
checkouts' ignored `results/` directories, including:

```text
results/vllm-w4a16-llama2-13b/
results/spinquant-isambard-llama2-13b/
results/vllm-w4afp8-llama2-13b/
results/vllm-spinquant-w4a16-llama2-13b/
results/vllm-spinquant-w4a16-boolq-llama2-13b/
results/vllm-spinquant-w4a16-serving-llama2-13b/
results/sglang-vllm-quarot-w4a16-serving/
```

Their reviewed job IDs, result hashes, and claim boundaries are indexed in
[`docs/EVIDENCE_LEDGER.md`](../docs/EVIDENCE_LEDGER.md). Do not copy large raw
results or checkpoints into Git merely to complete the index. Recent SpinQuant
W4A16 and SGLang-vLLM formal artifacts were re-read and rehashed during
acceptance. The remaining historical-row closeout action is a read-only
existence and SHA-256 recheck on the producing server; it must not rerun any
experiment when an artifact is missing.
