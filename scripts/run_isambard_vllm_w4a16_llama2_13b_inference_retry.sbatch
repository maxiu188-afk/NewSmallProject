#!/bin/bash
# Reuse accepted Llama-2-13B exports and retry only matched vLLM inference.
# Each complete model runs in a fresh Python process so vLLM releases its GPU
# reservation before the next model starts.
#SBATCH --job-name=vllm-13b-infer-retry
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=results/vllm-w4a16-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4a16-llama2-13b/%x-%j.err

set -euo pipefail

echo "VLLM_13B_RETRY_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
config="${project_root}/configs/deployment/vllm_w4a16_llama2_13b_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
calibration_manifest="${artifact_root}/calibration-128x2048/manifest.json"
unrotated_dir="${artifact_root}/unrotated-w4a16"
rotated_dir="${artifact_root}/rotated-w4a16"
results_dir="${project_root}/results/vllm-w4a16-llama2-13b"
unrotated_report="${results_dir}/unrotated-export.json"
rotated_report="${results_dir}/rotated-export.json"
inference_report="${results_dir}/offline-inference-retry.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_13B_RETRY_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${calibration_manifest}" \
    "${unrotated_report}" \
    "${rotated_report}" \
    "${unrotated_dir}/config.json" \
    "${rotated_dir}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_13B_RETRY_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_13B_RETRY_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi

echo "VLLM_13B_RETRY_STAGE=artifact_compatibility"
artifact_revision="$("${serving_env}/bin/python" - \
    "${unrotated_report}" "${rotated_report}" <<'PY'
import json
import sys
from pathlib import Path

reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
revisions = {report["project_revision"] for report in reports}
assert len(revisions) == 1, revisions
print(revisions.pop())
PY
)"
git -C "${project_root}" cat-file -e "${artifact_revision}^{commit}"
if ! git -C "${project_root}" merge-base --is-ancestor "${artifact_revision}" "${git_revision}"; then
    echo "VLLM_13B_RETRY_ARTIFACT_REVISION_NOT_ANCESTOR=${artifact_revision}" >&2
    exit 1
fi
export_inputs=(
    configs/deployment/vllm_w4a16_llama2_13b_isambard.json
    repro/offline_llama_rotation.py
    repro/structured_hadamard.py
    scripts/export_vllm_w4a16_llama2_13b.py
    scripts/prepare_vllm_w4a16_llama2_calibration.py
)
if ! git -C "${project_root}" diff --quiet \
    "${artifact_revision}..${git_revision}" -- "${export_inputs[@]}"; then
    echo "VLLM_13B_RETRY_EXPORT_INPUTS_CHANGED" >&2
    git -C "${project_root}" diff --name-only \
        "${artifact_revision}..${git_revision}" -- "${export_inputs[@]}" >&2
    exit 1
fi

"${serving_env}/bin/python" - \
    "${config}" \
    "${calibration_manifest}" \
    "${unrotated_report}" \
    "${rotated_report}" \
    "${unrotated_dir}" \
    "${rotated_dir}" \
    "${artifact_revision}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config, calibration, unrotated_report, rotated_report, unrotated, rotated = map(
    Path, sys.argv[1:7]
)
artifact_revision = sys.argv[7]
config_payload = json.loads(config.read_text(encoding="utf-8"))
reports = {
    "unrotated": json.loads(unrotated_report.read_text(encoding="utf-8")),
    "rotated": json.loads(rotated_report.read_text(encoding="utf-8")),
}
config_sha = sha256(config)
calibration_sha = sha256(calibration)
for mode, report in reports.items():
    assert report["status"] == "passed", report
    assert report["mode"] == mode, report
    assert report["project_revision"] == artifact_revision, report
    assert report["packed_decoder_linear_count"] == 280, report
    assert report["config_sha256"] == config_sha, report
    assert report["calibration_manifest_sha256"] == calibration_sha, report
assert reports["rotated"]["rotation"]["residual_hadamard_kind"] == "structured"
assert reports["rotated"]["prequant_rotation_max_absolute_logit_error"] <= float(
    config_payload["rotation"]["max_absolute_logit_error_tolerance"]
)
for checkpoint in (unrotated, rotated):
    packed = 0
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            packed += sum(key.endswith(".weight_packed") for key in handle.keys())
    assert packed == 280, (checkpoint, packed)
print(f"VLLM_13B_RETRY_ACCEPTED_ARTIFACT_REVISION={artifact_revision}")
PY

mkdir -p "${results_dir}"
echo "VLLM_13B_RETRY_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'artifact_revision=%s\n' "${artifact_revision}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    sha256sum \
        "${config}" \
        "${project_root}/scripts/run_isambard_vllm_w4a16_llama2_13b_inference_retry.sbatch" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_gate.py" \
        "${calibration_manifest}" \
        "${unrotated_report}" \
        "${rotated_report}"
} > "${results_dir}/inference-retry-source-manifest.txt"

echo "VLLM_13B_RETRY_STAGE=offline_inference"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4a16_llama2_13b_gate.py" \
    --config "${config}" \
    --original "${model_snapshot}" \
    --unrotated "${unrotated_dir}" \
    --rotated "${rotated_dir}" \
    --output "${inference_report}"

echo "VLLM_13B_RETRY_STAGE=assert_result"
"${serving_env}/bin/python" - \
    "${unrotated_report}" \
    "${rotated_report}" \
    "${inference_report}" \
    "${results_dir}/offline-inference-retry-models" <<'PY'
import json
import sys
from pathlib import Path

unrotated, rotated, inference = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:4]
]
worker_dir = Path(sys.argv[4])
assert unrotated["status"] == "passed", unrotated
assert rotated["status"] == "passed", rotated
assert unrotated["packed_decoder_linear_count"] == 280, unrotated
assert rotated["packed_decoder_linear_count"] == 280, rotated
assert rotated["rotation"]["residual_hadamard_kind"] == "structured", rotated
assert inference["status"] == "passed", inference
expected = {"bf16", "unrotated_w4a16", "rotated_w4a16"}
assert set(inference["models"]) == expected, inference["models"]
for name in expected:
    worker = json.loads((worker_dir / f"{name}.json").read_text(encoding="utf-8"))
    assert worker["name"] == name, worker
print("ISAMBARD_VLLM_W4A16_LLAMA2_13B_INFERENCE_RETRY_PASSED")
print("ISAMBARD_VLLM_W4A16_LLAMA2_13B_GATE_PASSED")
PY
