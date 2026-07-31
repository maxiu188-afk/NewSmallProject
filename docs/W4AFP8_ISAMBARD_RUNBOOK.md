# W4AFP8 Isambard preparation and runbook

## Current state

The QuaRot-only real-deployment gate is ready to submit independently. This
work does not affect the running corrected SpinQuant fake-quant chain because:

- it uses a separate Isambard checkout and Slurm allocation;
- W4AFP8 uses a separate
  `${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8/` artifact root;
- the QuaRot-only mode neither reads nor exports a SpinQuant rotation;
- fake-quant configs, outputs, and job scripts are unchanged.

The prepared path covers all later evidence gates:

| Gate | Prepared entry point | Runtime evidence |
|---|---|---|
| Backend capability | `scripts/audit_vllm_w4afp8_backend.py` | GH200 selects `CutlassW4A8LinearKernel` for every Llama-2-13B shape and rejects runtime `g_idx` |
| Export and load | `scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch` | Three separate packed checkpoints, 280 decoder linears each, fresh-process vLLM inference |
| Deployed accuracy | `scripts/run_isambard_vllm_w4afp8_llama2_13b_ppl.sbatch` | BF16 plus three W4AFP8 checkpoints on the retained 162 x 2048 WikiText-2 tokens |
| Service and acceleration | `scripts/run_isambard_vllm_w4afp8_llama2_13b_serving.sbatch` | Kernel log proof, endpoint smoke, concurrency-1/8 latency and throughput, memory and recovery |

The official LLM Compressor `W4AFP8` preset is group-128 symmetric INT4
weights plus symmetric dynamic per-token FP8 activations. The pinned vLLM
kernel requires Hopper SM90, FP8 E4M3 activations, no zero points, no runtime
activation-order `g_idx`, K/N divisible by 128, and BF16 output. These are
encoded as validation failures rather than documentation-only assumptions.

Official references:

- [LLM Compressor scheme selection](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/)
- [vLLM quantization support](https://docs.vllm.ai/en/stable/features/quantization/)
- [LLM Compressor repository and W4AFP8 examples](https://github.com/vllm-project/llm-compressor)
- [vLLM repository](https://github.com/vllm-project/vllm)

## QuaRot execution sequence

1. Run a scheduler preflight and then the isolated export/load gate. The job
   materializes the same-revision calibration artifact before export:

   ```bash
   sbatch --test-only \
     scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch quarot
   sbatch scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch quarot
   ```

2. Inspect the terminal state, exit code, stage markers, capability report,
   two export reports, offline-inference result, and checkpoint provenance.
   Only after acceptance, replace the pending `source_gate` fields in the PPL
   and serving configs with the gate job ID, relative result paths, and SHA-256
   hashes.
3. After Gate 1 acceptance, freeze its paths and hashes into QuaRot-only PPL
   and serving configs, then submit smoke before each formal job. The currently
   prepared PPL and serving configs still describe the later four-model joint
   study and must not be submitted against a three-model QuaRot gate.

The main PPL and serving jobs depend on accepted smoke results. They do not use
smoke outcomes to change the formal protocol.

## Result boundaries

- The current INT8 W4A8 fake-quant result is not W4AFP8 quality evidence.
- Backend capability selection is not checkpoint correctness, PPL, or speed.
- Fixed-token inference is not full held-out accuracy.
- Deployed PPL is not acceleration.
- Acceleration requires the matched full-model service result plus the selected
  W4AFP8 kernel pattern in every quantized server log.
- The initial SpinQuant checkpoint remains labelled `INT8-trained rotation
  transfer to W4AFP8`. If it is not competitive with unrotated W4AFP8, the
  accepted SpinQuant endpoint requires FP8-targeted rotation learning.
