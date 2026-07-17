# Generated result layout

`results/` contains generated artifacts and is ignored by default. Keep raw
model manifests, dataset cache references, logs, and JSON metrics on the
machine or persistent volume that produced them; commit only a reviewed summary
to `docs/` after the result boundary and provenance have been checked.

For the active server session, the durable location is:

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
