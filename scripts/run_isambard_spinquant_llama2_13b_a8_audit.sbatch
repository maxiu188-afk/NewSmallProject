#!/bin/bash
# Two-sequence diagnostic of runtime A8 coverage and quantization error.
# This is intentionally independent of the accepted formal PPL result.
# Run as:
#   sbatch scripts/run_isambard_spinquant_llama2_13b_a8_audit.sbatch
#SBATCH --job-name=spinquant-13b-a8-audit
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=results/spinquant-isambard-llama2-13b/%x-%j.out
#SBATCH --error=results/spinquant-isambard-llama2-13b/%x-%j.err

set -euo pipefail

echo "SPINQUANT_A8_AUDIT_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
evaluation_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
config="${project_root}/configs/spinquant/llama2_13b_nohad_w4a8_fake_quant_ppl_isambard.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
spinquant_artifact_root="${SPINQUANT_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-paper-nohad-w16a8}"
quarot_artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
rotation_manifest="${spinquant_artifact_root}/rotations-100step-5848547/rotation-manifest.json"
gptq_calibration_dir="${quarot_artifact_root}/calibration-128x2048"
tokens_manifest="${quarot_artifact_root}/ppl-wikitext2-test-162x2048/manifest.json"
results_dir="${project_root}/results/spinquant-isambard-llama2-13b"
output="${results_dir}/a8-coverage-audit-${SLURM_JOB_ID}.json"
max_sequences=2
max_calibration_sequences=8

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-spinquant-a8-audit-${SLURM_JOB_ID}"
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

echo "SPINQUANT_A8_AUDIT_STAGE=inputs"
for required in \
    "${evaluation_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${rotation_manifest}" \
    "${gptq_calibration_dir}/manifest.json" \
    "${gptq_calibration_dir}/dataset" \
    "${tokens_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SPINQUANT_A8_AUDIT_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SPINQUANT_A8_AUDIT_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "SPINQUANT_A8_AUDIT_STAGE=provenance_gate"
"${evaluation_env}/bin/python" - \
    "${config}" "${rotation_manifest}" "${gptq_calibration_dir}/manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from repro.spinquant.artifacts import load_rotation_artifact


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rotation_manifest = Path(sys.argv[2])
calibration_manifest = Path(sys.argv[3])
source = config["learned_rotation"]
assert sha256(rotation_manifest) == source["rotation_manifest_sha256"]
loaded = load_rotation_artifact(rotation_manifest, maximum_orthogonality_error=1e-4)
assert loaded["manifest"]["file"]["sha256"] == source["rotation_safetensors_sha256"]
calibration = json.loads(calibration_manifest.read_text(encoding="utf-8"))
assert calibration["status"] == "passed"
assert sha256(calibration_manifest) == config["gptq_calibration"]["manifest_sha256"]
assert calibration["dataset"]["token_ids_sha256"] == config["gptq_calibration"]["token_ids_sha256"]
print("SPINQUANT_A8_AUDIT_PROVENANCE_ACCEPTED")
PY

echo "SPINQUANT_A8_AUDIT_STAGE=runtime"
"${evaluation_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import platform
import torch

assert platform.python_version() == "3.11.7"
assert torch.__version__ == "2.11.0+cu129"
assert torch.version.cuda == "12.9"
assert metadata.version("transformers") == "5.10.1"
assert metadata.version("datasets") == "5.0.0"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("SPINQUANT_A8_AUDIT_RUNTIME_ACCEPTED")
PY

echo "SPINQUANT_A8_AUDIT_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'max_sequences=%s\n' "${max_sequences}"
    printf 'max_calibration_sequences=%s\n' "${max_calibration_sequences}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    sha256sum \
        "${config}" \
        "${project_root}/repro/gptq.py" \
        "${project_root}/repro/qdq.py" \
        "${project_root}/repro/spinquant/activation_audit.py" \
        "${project_root}/repro/spinquant/offline.py" \
        "${project_root}/scripts/spinquant/audit_llama_nohad_w4a8_activations.py" \
        "${project_root}/scripts/spinquant/evaluate_llama_nohad_w4a8_ppl.py" \
        "${project_root}/scripts/run_isambard_spinquant_llama2_13b_a8_audit.sbatch" \
        "${rotation_manifest}" \
        "${gptq_calibration_dir}/manifest.json" \
        "${tokens_manifest}"
} > "${results_dir}/a8-coverage-audit-${SLURM_JOB_ID}-source-manifest.txt"

echo "SPINQUANT_A8_AUDIT_STAGE=evaluate"
"${evaluation_env}/bin/python" \
    "${project_root}/scripts/spinquant/audit_llama_nohad_w4a8_activations.py" \
    --config "${config}" \
    --tokens-manifest "${tokens_manifest}" \
    --gptq-calibration-dir "${gptq_calibration_dir}" \
    --model-snapshot "${model_snapshot}" \
    --rotation-manifest "${rotation_manifest}" \
    --max-sequences "${max_sequences}" \
    --max-calibration-sequences "${max_calibration_sequences}" \
    --output "${output}"

echo "SPINQUANT_A8_AUDIT_STAGE=assert_result"
"${evaluation_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["evaluated_sequences"] == 2, result
assert result["scored_tokens"] == 4094, result
assert result["gptq_calibration_sequences"] == 8, result
assert set(result["cases"]) == {"unrotated_w4a8", "spinquant_nohad_w4a8"}, result
for case in result["cases"].values():
    assert case["preparation"]["gptq"]["linear_layers"] == 280, case
    assert set(case["modes"]) == {"current_repo", "paper_aligned"}, case
    for mode in case["modes"].values():
        assert mode["tokens"] == 4094, mode
        assert math.isfinite(mode["perplexity"]) and mode["perplexity"] > 0, mode
        audit = mode["activation_audit"]
        assert audit["registered_modules"] == 280, audit
        assert audit["called_modules"] == 280, audit
        assert audit["missing_modules"] == [], audit
        assert set(audit["role_module_counts"].values()) == {40}, audit
        assert audit["global"]["calls"] == 280, audit
        assert math.isfinite(audit["global"]["sqnr_db"]), audit
print("SPINQUANT_A8_AUDIT_RESULT_ACCEPTED")
PY

echo "SPINQUANT_A8_AUDIT_RESULT=${output}"
echo "ISAMBARD_SPINQUANT_LLAMA2_13B_A8_AUDIT_PASSED"
