#!/bin/bash
# Run BF16, unrotated GPTQ W4A8KV16, and learned SpinQuant no-had
# GPTQ W4A8KV16 fake-quant PPL through standard Transformers.
# Run as:
#   sbatch scripts/run_isambard_spinquant_llama2_13b_nohad_w4a8_ppl.sbatch smoke
#   sbatch --dependency=afterok:<smoke-job> \
#     scripts/run_isambard_spinquant_llama2_13b_nohad_w4a8_ppl.sbatch formal
#SBATCH --job-name=spinquant-13b-w4a8-ppl
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=results/spinquant-isambard-llama2-13b/%x-%j.out
#SBATCH --error=results/spinquant-isambard-llama2-13b/%x-%j.err

set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "smoke" && "${mode}" != "formal" ]]; then
    echo "usage: sbatch $0 {smoke|formal}" >&2
    exit 2
fi

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
evaluation_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
config="${project_root}/configs/spinquant/llama2_13b_nohad_w4a8_fake_quant_ppl_isambard.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
spinquant_artifact_root="${SPINQUANT_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-paper-nohad-w16a8}"
quarot_artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
training_job="5848547"
training_result="${project_root}/results/spinquant-isambard-llama2-13b/spinquant-13b-w16a8-100step-${training_job}.json"
rotation_manifest="${spinquant_artifact_root}/rotations-100step-${training_job}/rotation-manifest.json"
gptq_calibration_dir="${quarot_artifact_root}/calibration-128x2048"
tokens_manifest="${quarot_artifact_root}/ppl-wikitext2-test-162x2048/manifest.json"
results_dir="${project_root}/results/spinquant-isambard-llama2-13b"
smoke_result="${results_dir}/nohad-w4a8-fake-quant-ppl-smoke.json"
formal_result="${results_dir}/nohad-w4a8-fake-quant-ppl-formal.json"

if [[ "${mode}" == "smoke" ]]; then
    output="${smoke_result}"
    max_sequences=2
    max_calibration_sequences=8
    expected_tokens=4094
else
    output="${formal_result}"
    max_sequences=162
    max_calibration_sequences=128
    expected_tokens=331614
fi

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-spinquant-w4a8-ppl-${SLURM_JOB_ID}"
mkdir -p "${SCRATCHDIR}"
export TMPDIR="${SCRATCHDIR}"
export TMP="${SCRATCHDIR}"
export TEMP="${SCRATCHDIR}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=inputs"
for required in \
    "${evaluation_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${training_result}" \
    "${rotation_manifest}" \
    "${gptq_calibration_dir}/manifest.json" \
    "${gptq_calibration_dir}/dataset" \
    "${tokens_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SPINQUANT_NOHAD_W4A8_PPL_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" ]]; then
    echo "SPINQUANT_NOHAD_W4A8_PPL_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SPINQUANT_NOHAD_W4A8_PPL_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=provenance_gate"
"${evaluation_env}/bin/python" - \
    "${config}" "${training_result}" "${rotation_manifest}" \
    "${gptq_calibration_dir}/manifest.json" "${git_revision}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from repro.spinquant.artifacts import load_rotation_artifact


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result_path = Path(sys.argv[2])
rotation_manifest = Path(sys.argv[3])
calibration_manifest_path = Path(sys.argv[4])
revision = sys.argv[5]

source = config["learned_rotation"]
result = json.loads(result_path.read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["slurm_job_id"] == source["training_job_id"], result
assert result["project_revision"] == source["project_revision"], result
assert sha256(result_path) == source["training_result_sha256"], result
assert sha256(rotation_manifest) == source["rotation_manifest_sha256"], result
assert Path(result["rotation_manifest"]).resolve() == rotation_manifest.resolve(), result
assert result["rotation_safetensors_sha256"] == source["rotation_safetensors_sha256"], result
assert subprocess.run(
    ["git", "merge-base", "--is-ancestor", result["project_revision"], revision],
    check=False,
).returncode == 0, result
loaded = load_rotation_artifact(
    rotation_manifest,
    maximum_orthogonality_error=1e-4,
)
assert loaded["manifest"]["file"]["sha256"] == source["rotation_safetensors_sha256"], loaded

calibration_source = config["gptq_calibration"]
calibration = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
assert calibration["status"] == "passed", calibration
assert sha256(calibration_manifest_path) == calibration_source["manifest_sha256"], calibration
dataset = calibration["dataset"]
assert dataset["token_ids_sha256"] == calibration_source["token_ids_sha256"], dataset
assert dataset["samples"] == calibration_source["samples"] == 128, dataset
assert dataset["sequence_length"] == calibration_source["sequence_length"] == 2048, dataset
assert dataset["dataset_revision"] == calibration_source["dataset_revision"], dataset
print("SPINQUANT_NOHAD_W4A8_PPL_PROVENANCE_ACCEPTED")
PY

if [[ "${mode}" == "formal" ]]; then
    echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=smoke_dependency"
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "SPINQUANT_NOHAD_W4A8_PPL_FORMAL_REQUIRES_AFTEROK_DEPENDENCY" >&2
        exit 1
    fi
    "${evaluation_env}/bin/python" - \
        "${smoke_result}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert smoke["status"] == "passed", smoke
assert smoke["project_revision"] == sys.argv[2], smoke
assert smoke["evaluated_sequences"] == 2, smoke
assert smoke["scored_tokens"] == 4094, smoke
assert smoke["gptq_calibration_sequences"] == 8, smoke
assert set(smoke["models"]) == {
    "bf16",
    "unrotated_w4a8",
    "spinquant_nohad_w4a8",
}, smoke
print("SPINQUANT_NOHAD_W4A8_PPL_SMOKE_DEPENDENCY_ACCEPTED")
PY
fi

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=runtime"
"${evaluation_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import platform
import torch

assert platform.python_version() == "3.11.7", platform.python_version()
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("transformers") == "5.10.1"
assert metadata.version("datasets") == "5.0.0"
assert metadata.version("safetensors") == "0.8.0"
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("SPINQUANT_NOHAD_W4A8_PPL_RUNTIME_ACCEPTED")
PY

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=source_manifest"
{
    printf 'mode=%s\n' "${mode}"
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'accepted_training_job=%s\n' "${training_job}"
    printf 'slurm_dependency=%s\n' "${SLURM_JOB_DEPENDENCY:-unset}"
    printf 'spinquant_artifact_root=%s\n' "${spinquant_artifact_root}"
    printf 'gptq_calibration_dir=%s\n' "${gptq_calibration_dir}"
    printf 'evaluation_token_root=%s\n' "${quarot_artifact_root}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    sha256sum \
        "${config}" \
        "${project_root}/repro/gptq.py" \
        "${project_root}/repro/qdq.py" \
        "${project_root}/repro/spinquant/artifacts.py" \
        "${project_root}/repro/spinquant/offline.py" \
        "${project_root}/repro/spinquant/rotations.py" \
        "${project_root}/scripts/spinquant/evaluate_llama_w4a16_ppl.py" \
        "${project_root}/scripts/spinquant/evaluate_llama_nohad_w4a8_ppl.py" \
        "${project_root}/scripts/run_isambard_spinquant_llama2_13b_nohad_w4a8_ppl.sbatch" \
        "${model_snapshot}/config.json" \
        "${training_result}" \
        "${rotation_manifest}" \
        "${gptq_calibration_dir}/manifest.json" \
        "${tokens_manifest}"
} > "${results_dir}/nohad-w4a8-fake-quant-ppl-${mode}-source-manifest.txt"

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=evaluate"
"${evaluation_env}/bin/python" \
    "${project_root}/scripts/spinquant/evaluate_llama_nohad_w4a8_ppl.py" \
    --config "${config}" \
    --tokens-manifest "${tokens_manifest}" \
    --gptq-calibration-dir "${gptq_calibration_dir}" \
    --model-snapshot "${model_snapshot}" \
    --rotation-manifest "${rotation_manifest}" \
    --max-sequences "${max_sequences}" \
    --max-calibration-sequences "${max_calibration_sequences}" \
    --output "${output}"

echo "SPINQUANT_NOHAD_W4A8_PPL_STAGE=assert_result"
"${evaluation_env}/bin/python" - \
    "${output}" "${git_revision}" "${max_sequences}" \
    "${max_calibration_sequences}" "${expected_tokens}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["evaluated_sequences"] == int(sys.argv[3]), result
assert result["gptq_calibration_sequences"] == int(sys.argv[4]), result
assert result["scored_tokens"] == int(sys.argv[5]), result
expected = {"bf16", "unrotated_w4a8", "spinquant_nohad_w4a8"}
assert set(result["models"]) == expected, result
for name, model in result["models"].items():
    assert model["tokens"] == int(sys.argv[5]), model
    assert math.isfinite(model["mean_nll"]), model
    assert math.isfinite(model["perplexity"]) and model["perplexity"] > 0, model
    preparation = model["preparation"]
    if name == "bf16":
        assert preparation["gptq"] is None, model
        assert preparation["activation_quantization"] is None, model
        assert preparation["rotation"] is None, model
        continue
    assert preparation["gptq"]["layers"] == 40, model
    assert preparation["gptq"]["linear_layers"] == 280, model
    assert preparation["gptq"]["calibration_sequences"] == int(sys.argv[4]), model
    activation = preparation["activation_quantization"]
    assert activation["quantized_decoder_linears"] == 280, model
    assert activation["activation_bits"] == 8, model
    assert activation["activation_symmetric"] is False, model
    assert activation["activation_granularity"] == "per_token_last_axis", model
    assert activation["quantized_lm_head"] is False, model
    if name == "spinquant_nohad_w4a8":
        assert preparation["rotation"]["online_rotation_modules"] == 0, model
    else:
        assert preparation["rotation"] is None, model
print("SPINQUANT_NOHAD_W4A8_PPL_RESULT_ACCEPTED")
PY

if [[ "${mode}" == "smoke" ]]; then
    echo "ISAMBARD_SPINQUANT_LLAMA2_13B_NOHAD_W4A8_PPL_SMOKE_PASSED"
else
    echo "ISAMBARD_SPINQUANT_LLAMA2_13B_NOHAD_W4A8_PPL_FORMAL_PASSED"
fi
