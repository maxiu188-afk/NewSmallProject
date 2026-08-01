#!/bin/bash
# Run as:
#   sbatch scripts/run_isambard_vllm_w4a16_llama2_13b_ppl.sbatch smoke
#   sbatch --dependency=afterok:<smoke-job> \
#     scripts/run_isambard_vllm_w4a16_llama2_13b_ppl.sbatch formal
#SBATCH --job-name=vllm-13b-ppl
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=results/vllm-w4a16-ppl-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4a16-ppl-llama2-13b/%x-%j.err

set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "smoke" && "${mode}" != "formal" ]]; then
    echo "usage: sbatch $0 {smoke|formal}" >&2
    exit 2
fi

echo "VLLM_13B_PPL_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
config="${project_root}/configs/deployment/vllm_w4a16_llama2_13b_ppl_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
unrotated_dir="${artifact_root}/unrotated-w4a16"
rotated_dir="${artifact_root}/rotated-w4a16"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
source_result="${project_root}/results/vllm-w4a16-llama2-13b/offline-inference-retry.json"
tokens_dir="${artifact_root}/ppl-wikitext2-test-162x2048"
tokens_manifest="${tokens_dir}/manifest.json"
results_dir="${project_root}/results/vllm-w4a16-ppl-llama2-13b"
smoke_result="${results_dir}/ppl-smoke.json"
formal_result="${results_dir}/ppl-formal.json"

if [[ "${mode}" == "smoke" ]]; then
    output="${smoke_result}"
    max_sequences=2
    expected_tokens=4094
else
    output="${formal_result}"
    max_sequences=162
    expected_tokens=331614
fi

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-ppl-${SLURM_JOB_ID}"
mkdir -p "${SCRATCHDIR}"
export TMPDIR="${SCRATCHDIR}"
export TMP="${SCRATCHDIR}"
export TEMP="${SCRATCHDIR}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_13B_PPL_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${unrotated_dir}/config.json" \
    "${rotated_dir}/config.json" \
    "${source_result}" \
    "${tokens_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_13B_PPL_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_13B_PPL_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_13B_PPL_STAGE=source_gate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/validate_vllm_w4a16_llama2_13b_serving_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4a16=${unrotated_dir}" \
    --model "rotated_w4a16=${rotated_dir}"

echo "VLLM_13B_PPL_STAGE=token_manifest"
"${serving_env}/bin/python" - \
    "${config}" "${tokens_manifest}" "${git_revision}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config_path, manifest_path = map(Path, sys.argv[1:3])
revision = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["status"] == "passed", manifest
assert manifest["config_sha256"] == sha256(config_path), manifest
assert manifest["dataset"]["materialized_samples"] == 162, manifest
assert manifest["dataset"]["scored_tokens"] == 331614, manifest
tokens_path = Path(manifest["tokens"]["path"])
assert sha256(tokens_path) == manifest["tokens"]["sha256"], manifest
ancestor = subprocess.run(
    ["git", "-C", str(config_path.parents[2]), "merge-base", "--is-ancestor",
     manifest["project_revision"], revision],
    check=False,
)
assert ancestor.returncode == 0, manifest
print("VLLM_13B_PPL_TOKEN_MANIFEST_ACCEPTED")
PY

if [[ "${mode}" == "formal" ]]; then
    echo "VLLM_13B_PPL_STAGE=smoke_dependency"
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "VLLM_13B_PPL_FORMAL_REQUIRES_AFTEROK_DEPENDENCY" >&2
        exit 1
    fi
    "${serving_env}/bin/python" - \
        "${smoke_result}" "${git_revision}" "${tokens_manifest}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


smoke_path, revision, manifest_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
assert smoke["status"] == "passed", smoke
assert smoke["project_revision"] == revision, smoke
assert smoke["tokens_manifest_sha256"] == sha256(manifest_path), smoke
assert smoke["evaluated_sequences"] == 2, smoke
assert smoke["scored_tokens"] == 4094, smoke
assert set(smoke["models"]) == {"bf16", "unrotated_w4a16", "rotated_w4a16"}, smoke
print("VLLM_13B_PPL_SMOKE_DEPENDENCY_ACCEPTED")
PY
fi

echo "VLLM_13B_PPL_STAGE=source_manifest"
{
    printf 'mode=%s\n' "${mode}"
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'source_gate_job=5769503\n'
    printf 'slurm_dependency=%s\n' "${SLURM_JOB_DEPENDENCY:-unset}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    sha256sum \
        "${config}" \
        "${project_root}/scripts/run_isambard_vllm_w4a16_llama2_13b_ppl.sbatch" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_ppl.py" \
        "${project_root}/scripts/prepare_vllm_w4a16_llama2_ppl_tokens.py" \
        "${project_root}/scripts/validate_vllm_w4a16_llama2_13b_serving_inputs.py" \
        "${source_result}" \
        "${tokens_manifest}"
} > "${results_dir}/ppl-${mode}-source-manifest.txt"

echo "VLLM_13B_PPL_STAGE=runtime"
"${serving_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import torch

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("vllm") == "0.25.1+cu129"
assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("VLLM_13B_PPL_RUNTIME_ACCEPTED")
PY

echo "VLLM_13B_PPL_STAGE=evaluate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4a16_llama2_13b_ppl.py" \
    --config "${config}" \
    --tokens-manifest "${tokens_manifest}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4a16=${unrotated_dir}" \
    --model "rotated_w4a16=${rotated_dir}" \
    --max-sequences "${max_sequences}" \
    --output "${output}"

echo "VLLM_13B_PPL_STAGE=assert_result"
"${serving_env}/bin/python" - \
    "${output}" "${git_revision}" "${max_sequences}" "${expected_tokens}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["evaluated_sequences"] == int(sys.argv[3]), result
assert result["scored_tokens"] == int(sys.argv[4]), result
expected = {"bf16", "unrotated_w4a16", "rotated_w4a16"}
assert set(result["models"]) == expected, result
for name, model in result["models"].items():
    assert model["tokens"] == int(sys.argv[4]), model
    assert math.isfinite(model["mean_nll"]), model
    assert math.isfinite(model["perplexity"]) and model["perplexity"] > 0, model
    expected_packed = 0 if name == "bf16" else 280
    assert model["packed_decoder_linear_count"] == expected_packed, model
print("VLLM_13B_PPL_RESULT_ACCEPTED")
PY

if [[ "${mode}" == "smoke" ]]; then
    echo "ISAMBARD_VLLM_W4A16_LLAMA2_13B_PPL_SMOKE_PASSED"
else
    echo "ISAMBARD_VLLM_W4A16_LLAMA2_13B_PPL_FORMAL_PASSED"
fi
