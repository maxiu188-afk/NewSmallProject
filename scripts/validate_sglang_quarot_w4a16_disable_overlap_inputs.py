#!/usr/bin/env python3
"""Bind the SGLang no-overlap third arm to formal job 5960180."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


EXPECTED_CASES = [
    {"name": "latency_c1", "max_concurrency": 1},
    {"name": "throughput_c8", "max_concurrency": 8},
]
EXPECTED_REFERENCE_CASES = {
    "vllm:latency_c1",
    "sglang:latency_c1",
    "vllm:throughput_c8",
    "sglang:throughput_c8",
}
EXPECTED_PROTOCOL = {
    "protocol": "repository_openai_completions_stream_v1",
    "endpoint": "/v1/completions",
    "temperature": 0.0,
    "repetition_penalty": 1.0,
    "ignore_eos": True,
    "stream_include_usage": True,
    "validation_requests": 1,
    "warmup_requests": 4,
    "measured_requests": 64,
    "repetitions": 3,
    "metric_percentiles": [50, 95, 99],
    "memory_sample_interval_seconds": 0.2,
}
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_git_blob(
    project_root: Path, revision: str, relative_path: Path
) -> str | None:
    if GIT_REVISION.fullmatch(revision) is None:
        raise RuntimeError("reference manifest Git revision is malformed")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("reference manifest Git path escapes the project root")
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "show",
            f"{revision}:{relative_path.as_posix()}",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def _require_git_revision(project_root: Path, revision: str) -> None:
    if GIT_REVISION.fullmatch(revision) is None:
        raise RuntimeError("reference manifest Git revision is malformed")
    completed = subprocess.run(
        ["git", "-C", str(project_root), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("reference manifest Git revision is unavailable")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_benchmark(benchmark: dict[str, Any]) -> None:
    for key, expected in EXPECTED_PROTOCOL.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(f"benchmark field drifted: {key}")
    if benchmark.get("cases") != EXPECTED_CASES:
        raise RuntimeError("concurrency matrix drifted")


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("status") != "authorized_not_submitted":
        raise RuntimeError("no-overlap third arm is not authorized")
    experiment = spec.get("experiment", {})
    expected_experiment = {
        "backend": "sglang",
        "variant": "disable_overlap_schedule",
        "server_argument": "--disable-overlap-schedule",
        "comparison_design": (
            "protocol_matched_historical_third_arm_not_paired_with_reference_job"
        ),
        "only_intended_runtime_change": "disable SGLang overlap scheduling",
    }
    if experiment != expected_experiment:
        raise RuntimeError("no-overlap experiment definition drifted")
    reference = spec.get("reference_formal_job", {})
    if reference.get("job_id") != "5960180":
        raise RuntimeError("third arm is not bound to formal job 5960180")
    if reference.get("comparison_status") != "formal_matrix_complete":
        raise RuntimeError("reference comparison status drifted")
    _validate_benchmark(spec.get("benchmark", {}))


def _validate_base_config(base: dict[str, Any]) -> None:
    if base.get("model") != "quarot_w4a16":
        raise RuntimeError("base checkpoint route drifted")
    if base.get("backends") != ["vllm", "sglang"]:
        raise RuntimeError("base backend capability set drifted")
    server = base.get("server", {})
    expected_server = {
        "host": "127.0.0.1",
        "port": 18000,
        "dtype": "bfloat16",
        "max_model_len": 512,
        "max_running_requests": 8,
        "seed": 0,
        "kv_cache_dtype": "bfloat16",
        "kv_cache_memory_bytes": 8 * 1024**3,
        "max_total_tokens": 10485,
        "disable_prefix_cache": True,
        "disable_chunked_prefill": True,
    }
    for key, expected in expected_server.items():
        if server.get(key) != expected:
            raise RuntimeError(f"base server field drifted: {key}")
    corpus = base.get("request_corpus", {})
    if corpus.get("file_sha256") != (
        "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1"
    ):
        raise RuntimeError("base request corpus drifted")
    if corpus.get("requests") != 64:
        raise RuntimeError("base request count drifted")
    sglang = base.get("sglang", {})
    if sglang.get("expected_version") != "0.5.16":
        raise RuntimeError("SGLang version drifted")
    if sglang.get("attention_backend") != "flashinfer":
        raise RuntimeError("SGLang attention backend drifted")
    if sglang.get("offline_quantization_argument") is not None:
        raise RuntimeError("offline quantization override drifted")
    if sglang.get("enable_jit_deep_gemm") is not False:
        raise RuntimeError("JIT DeepGEMM policy drifted")


def _validate_reference_result(
    result: dict[str, Any], reference: dict[str, Any]
) -> None:
    expected_scalars = {
        "status": "passed",
        "comparison_status": "formal_matrix_complete",
        "project_revision": reference["project_revision"],
        "config_sha256": reference["effective_config_sha256"],
        "model": "quarot_w4a16",
    }
    for key, expected in expected_scalars.items():
        if result.get(key) != expected:
            raise RuntimeError(f"reference formal result drifted: {key}")
    if result.get("request_corpus", {}).get("sha256") != reference[
        "request_corpus_sha256"
    ]:
        raise RuntimeError("reference request corpus drifted")
    _validate_benchmark(result.get("benchmark", {}))
    repetitions = result.get("repetitions", [])
    if len(repetitions) != 3:
        raise RuntimeError("reference repetition count drifted")
    for repetition in repetitions:
        cases = repetition.get("cases", {})
        if set(cases) != EXPECTED_REFERENCE_CASES:
            raise RuntimeError("reference case set drifted")
        for case in cases.values():
            metrics = case.get("metrics", {})
            if case.get("status") != "passed":
                raise RuntimeError("reference contains a failed cell")
            if metrics.get("completed") != 64 or metrics.get("failed") != 0:
                raise RuntimeError("reference cell request counts drifted")
            if metrics.get("input_tokens") != 64 * 256:
                raise RuntimeError("reference input-token count drifted")
            if metrics.get("output_tokens") != 64 * 64:
                raise RuntimeError("reference output-token count drifted")


def _validate_manifest(
    path: Path,
    *,
    project_root: Path,
    expected_revision: str,
    minimum_entries: int = 40,
) -> tuple[dict[str, str], int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key in {"git_revision", "git_status"}:
            metadata[key] = value
    if metadata.get("git_revision") != expected_revision:
        raise RuntimeError("reference manifest Git revision drifted")
    if metadata.get("git_status") != "clean":
        raise RuntimeError("reference manifest was not produced from a clean checkout")

    resolved_root = project_root.resolve()
    _require_git_revision(resolved_root, expected_revision)
    recorded: dict[str, str] = {}
    git_blob_entries = 0
    for line in lines:
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            continue
        expected, raw_path = match.groups()
        artifact = Path(raw_path)
        try:
            relative_path = artifact.resolve().relative_to(resolved_root)
        except ValueError:
            relative_path = None
        git_blob_sha256 = None
        if relative_path is not None:
            git_blob_sha256 = _sha256_git_blob(
                resolved_root, expected_revision, relative_path
            )
        if git_blob_sha256 is not None:
            observed = git_blob_sha256
            git_blob_entries += 1
        elif artifact.is_file():
            observed = _sha256(artifact)
        else:
            raise RuntimeError(f"reference manifest artifact is missing: {artifact}")
        if observed != expected:
            raise RuntimeError(f"reference manifest artifact drifted: {artifact}")
        recorded[str(artifact.resolve())] = observed
    if len(recorded) < minimum_entries:
        raise RuntimeError("reference source manifest is unexpectedly incomplete")
    return recorded, git_blob_entries


def validate(
    *,
    project_root: Path,
    spec_path: Path,
    reference_result_path: Path,
    reference_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _validate_spec(spec)
    base_path = (project_root / spec["base_config"]).resolve()
    if _sha256(base_path) != spec["base_config_sha256"]:
        raise RuntimeError("base serving config drifted")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    _validate_base_config(base)

    reference = spec["reference_formal_job"]
    if _sha256(reference_result_path) != reference["result_sha256"]:
        raise RuntimeError("reference formal result SHA-256 drifted")
    if _sha256(reference_manifest_path) != reference["source_manifest_sha256"]:
        raise RuntimeError("reference source-manifest SHA-256 drifted")
    result = json.loads(reference_result_path.read_text(encoding="utf-8"))
    _validate_reference_result(result, reference)
    manifest_entries, git_blob_entries = _validate_manifest(
        reference_manifest_path,
        project_root=project_root,
        expected_revision=reference["project_revision"],
    )
    if reference["result_sha256"] not in manifest_entries.values():
        raise RuntimeError("reference result is not bound by its source manifest")

    effective = copy.deepcopy(base)
    effective["config_version"] = 2
    effective["status"] = spec["status"]
    effective["reference_formal_job"] = copy.deepcopy(reference)
    effective["experiment"] = copy.deepcopy(spec["experiment"])
    effective["benchmark"] = copy.deepcopy(spec["benchmark"])
    effective["scope"] = spec["scope"]
    effective["formal_config_source"] = {
        "path": str(spec_path.resolve()),
        "sha256": _sha256(spec_path),
        "base_config": str(base_path),
        "base_config_sha256": _sha256(base_path),
    }
    validation = {
        "status": "accepted",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_formal_job_id": reference["job_id"],
        "reference_result_sha256": _sha256(reference_result_path),
        "reference_source_manifest_sha256": _sha256(reference_manifest_path),
        "reference_manifest_entries_rehashed": len(manifest_entries),
        "reference_manifest_file_entries_rehashed": (
            len(manifest_entries) - git_blob_entries
        ),
        "reference_manifest_git_blob_entries_rehashed": git_blob_entries,
        "request_corpus_sha256": reference["request_corpus_sha256"],
        "experiment": copy.deepcopy(spec["experiment"]),
        "spec_sha256": _sha256(spec_path),
        "base_config_sha256": _sha256(base_path),
    }
    return effective, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-source-manifest", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-record", type=Path, required=True)
    args = parser.parse_args()
    effective, validation = validate(
        project_root=args.project_root.resolve(),
        spec_path=args.config.resolve(),
        reference_result_path=args.reference_result.resolve(),
        reference_manifest_path=args.reference_source_manifest.resolve(),
    )
    _write_json(args.output_config.resolve(), effective)
    _write_json(args.output_record.resolve(), validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    print("SGLANG_QUAROT_W4A16_DISABLE_OVERLAP_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
