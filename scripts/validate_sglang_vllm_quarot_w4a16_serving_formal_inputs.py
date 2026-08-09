#!/usr/bin/env python3
"""Bind a formal serving matrix to the accepted cross-backend smoke evidence."""

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
EXPECTED_BACKEND_ORDERS = [
    ["vllm", "sglang"],
    ["sglang", "vllm"],
    ["vllm", "sglang"],
]
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_output(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_protocol(spec: dict[str, Any]) -> None:
    if spec.get("status") != "accepted_serving_smoke_bound_formal_matrix_authorized":
        raise RuntimeError("formal protocol is not authorized")
    gate = spec.get("serving_smoke_gate", {})
    if gate.get("status") != "accepted" or gate.get("job_id") != "5952554":
        raise RuntimeError("formal protocol is not bound to accepted smoke 5952554")
    benchmark = spec.get("benchmark", {})
    expected_scalars = {
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
    for key, expected in expected_scalars.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(f"formal benchmark field drifted: {key}")
    if benchmark.get("cases") != EXPECTED_CASES:
        raise RuntimeError("formal concurrency matrix drifted")
    if benchmark.get("backend_orders") != EXPECTED_BACKEND_ORDERS:
        raise RuntimeError("formal paired backend order drifted")


def _validate_manifest(path: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            continue
        expected, raw_path = match.groups()
        artifact = Path(raw_path)
        if not artifact.is_file():
            raise RuntimeError(f"accepted smoke manifest artifact is missing: {artifact}")
        observed = _sha256(artifact)
        if observed != expected:
            raise RuntimeError(f"accepted smoke manifest artifact drifted: {artifact}")
        recorded[str(artifact.resolve())] = observed
    if len(recorded) < 10:
        raise RuntimeError("accepted smoke source manifest is unexpectedly incomplete")
    return recorded


def validate(
    *,
    project_root: Path,
    formal_spec_path: Path,
    smoke_result_path: Path,
    smoke_request_corpus_path: Path,
    smoke_preflight_path: Path,
    smoke_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads(formal_spec_path.read_text(encoding="utf-8"))
    _validate_protocol(spec)
    gate = spec["serving_smoke_gate"]
    current_revision = _git_output(project_root, "rev-parse", "HEAD")
    source_revision = gate["source_revision"]
    if subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "merge-base",
            "--is-ancestor",
            source_revision,
            current_revision,
        ],
        check=False,
    ).returncode != 0:
        raise RuntimeError("accepted serving-smoke revision is not an ancestor")

    base_config_path = (project_root / spec["base_config"]).resolve()
    if _sha256(base_config_path) != spec["base_config_sha256"]:
        raise RuntimeError("accepted serving-smoke base config drifted")
    producer_hashes: dict[str, str] = {}
    producer_paths = list(gate["producer_sha256"])
    for relative, expected in gate["producer_sha256"].items():
        path = project_root / relative
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"accepted serving-smoke producer drifted: {relative}")
        producer_hashes[relative] = observed
    if subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "diff",
            "--quiet",
            f"{source_revision}..{current_revision}",
            "--",
            *producer_paths,
        ],
        check=False,
    ).returncode != 0:
        raise RuntimeError("accepted serving-smoke producer changed after its revision")

    artifacts = {
        "result": (smoke_result_path, gate["result_sha256"]),
        "request_corpus": (
            smoke_request_corpus_path,
            gate["request_corpus_sha256"],
        ),
        "sglang_preflight": (
            smoke_preflight_path,
            gate["sglang_preflight_sha256"],
        ),
        "source_manifest": (smoke_manifest_path, gate["source_manifest_sha256"]),
    }
    artifact_hashes = {}
    for name, (path, expected) in artifacts.items():
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"accepted serving-smoke {name} drifted")
        artifact_hashes[name] = observed

    manifest_entries = _validate_manifest(smoke_manifest_path)
    smoke_result = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    if smoke_result.get("status") != "recorded":
        raise RuntimeError("accepted serving smoke was not recorded")
    if smoke_result.get("comparison_status") != gate["expected_comparison_status"]:
        raise RuntimeError("accepted serving-smoke comparison did not pass")
    if smoke_result.get("project_revision") != source_revision:
        raise RuntimeError("accepted serving-smoke source revision drifted")
    if smoke_result.get("config_sha256") != spec["base_config_sha256"]:
        raise RuntimeError("accepted serving smoke used a different base config")
    if smoke_result.get("request_corpus", {}).get("sha256") != gate[
        "request_corpus_sha256"
    ]:
        raise RuntimeError("accepted serving smoke used a different request corpus")
    expected_smoke_cases = {"vllm:smoke_c1", "sglang:smoke_c1"}
    if set(smoke_result.get("cases", {})) != expected_smoke_cases:
        raise RuntimeError("accepted serving-smoke case set drifted")
    for case in smoke_result["cases"].values():
        metrics = case.get("metrics", {})
        if case.get("status") != "passed" or metrics.get("completed") != 8:
            raise RuntimeError("accepted serving-smoke case did not pass 8 requests")
        if metrics.get("failed") != 0:
            raise RuntimeError("accepted serving-smoke case recorded a failed request")

    smoke_preflight = json.loads(smoke_preflight_path.read_text(encoding="utf-8"))
    if smoke_preflight.get("status") != "passed":
        raise RuntimeError("accepted serving-smoke SGLang preflight did not pass")
    if smoke_preflight.get("config_sha256") != spec["base_config_sha256"]:
        raise RuntimeError("accepted serving-smoke preflight used another config")

    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if base_config.get("backends") != ["vllm", "sglang"]:
        raise RuntimeError("accepted serving-smoke backend set drifted")
    effective = copy.deepcopy(base_config)
    effective["config_version"] = 2
    effective["status"] = spec["status"]
    effective["serving_smoke_gate"] = copy.deepcopy(gate)
    effective["benchmark"] = copy.deepcopy(spec["benchmark"])
    effective["scope"] = spec["scope"]
    effective["formal_config_source"] = {
        "path": str(formal_spec_path.resolve()),
        "sha256": _sha256(formal_spec_path),
        "base_config": str(base_config_path),
        "base_config_sha256": _sha256(base_config_path),
    }
    validation = {
        "status": "accepted",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "current_revision": current_revision,
        "serving_smoke_job_id": gate["job_id"],
        "serving_smoke_source_revision": source_revision,
        "artifact_sha256": artifact_hashes,
        "manifest_entries_rehashed": len(manifest_entries),
        "producer_sha256": producer_hashes,
        "formal_spec_sha256": _sha256(formal_spec_path),
        "base_config_sha256": _sha256(base_config_path),
    }
    return effective, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--smoke-request-corpus", type=Path, required=True)
    parser.add_argument("--smoke-preflight", type=Path, required=True)
    parser.add_argument("--smoke-source-manifest", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-record", type=Path, required=True)
    args = parser.parse_args()
    effective, validation = validate(
        project_root=args.project_root.resolve(),
        formal_spec_path=args.formal_config.resolve(),
        smoke_result_path=args.smoke_result.resolve(),
        smoke_request_corpus_path=args.smoke_request_corpus.resolve(),
        smoke_preflight_path=args.smoke_preflight.resolve(),
        smoke_manifest_path=args.smoke_source_manifest.resolve(),
    )
    _write_json(args.output_config.resolve(), effective)
    _write_json(args.output_record.resolve(), validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    print("SGLANG_VLLM_QUAROT_W4A16_FORMAL_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
