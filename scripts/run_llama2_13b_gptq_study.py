#!/usr/bin/env python3
"""Run the ordered Llama-2-13B GPTQ comparison, ablation, and calibration study."""

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_RUNNER = PROJECT_ROOT / "scripts" / "run_quarot_pipeline.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "llama2-13b-wikitext2-gptq-study"


@dataclass(frozen=True)
class StudyRow:
    """One complete, independently restartable server experiment."""

    identifier: str
    config: str
    calibration_sequences: int
    purpose: str


# The first five rows form the component ablation at 128 calibration sequences.
# The last four complete the 32/64/128 calibration-size grid without rerunning
# either shared 128-sequence endpoint.
STUDY_ROWS: Tuple[StudyRow, ...] = (
    StudyRow("naive-f3-cal128", "llama2_13b_wikitext2_naive_w4a4_gptq.json", 128, "Unrotated GPTQ F3 control."),
    StudyRow("quarot-residual-cal128", "llama2_13b_wikitext2_quarot_residual_w4a4_gptq.json", 128, "Residual Hadamard only."),
    StudyRow("quarot-residual-vo-cal128", "llama2_13b_wikitext2_quarot_residual_vo_w4a4_gptq.json", 128, "Residual Hadamard plus V/O rotation."),
    StudyRow("quarot-residual-vo-mlp-cal128", "llama2_13b_wikitext2_quarot_residual_vo_mlp_w4a4_gptq.json", 128, "Residual, V/O, and MLP transforms."),
    StudyRow("quarot-f4-cal128", "llama2_13b_wikitext2_quarot_w4a4_gptq.json", 128, "Complete QuaRot F4 candidate, including post-RoPE Q/K."),
    StudyRow("naive-f3-cal32", "llama2_13b_wikitext2_naive_w4a4_gptq_cal32.json", 32, "Unrotated GPTQ calibration-size control."),
    StudyRow("quarot-f4-cal32", "llama2_13b_wikitext2_quarot_w4a4_gptq_cal32.json", 32, "Complete QuaRot calibration-size candidate."),
    StudyRow("naive-f3-cal64", "llama2_13b_wikitext2_naive_w4a4_gptq_cal64.json", 64, "Unrotated GPTQ calibration-size control."),
    StudyRow("quarot-f4-cal64", "llama2_13b_wikitext2_quarot_w4a4_gptq_cal64.json", 64, "Complete QuaRot calibration-size candidate."),
)


class StudyError(ValueError):
    """Raised when a requested study run would be incomplete or ambiguous."""


def select_rows(identifiers: Iterable[str]) -> Tuple[StudyRow, ...]:
    """Return rows in dependency order, rejecting unknown or duplicate selectors."""
    requested = tuple(identifiers)
    if not requested:
        return STUDY_ROWS
    if len(set(requested)) != len(requested):
        raise StudyError("--only may not name a row more than once")
    available = {row.identifier for row in STUDY_ROWS}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise StudyError("unknown study row(s): {}".format(", ".join(unknown)))
    selected = set(requested)
    return tuple(row for row in STUDY_ROWS if row.identifier in selected)


def _result_is_complete(path: Path, row: StudyRow) -> bool:
    """Accept only a finite successful JSON result for the exact calibration size."""
    try:
        with path.open(encoding="utf-8") as handle:
            result: Dict[str, Any] = json.load(handle)
        candidate = result["candidate"]
        calibration = result["calibration"]
        gptq = result["gptq"]
        metrics = (candidate["perplexity"], candidate["mean_nll"], candidate["tokens"])
        errors = (result["logit_error"]["mean_absolute"], result["logit_error"]["max_absolute"])
        return (
            result["weight_quantization"]["method"] == "gptq"
            and calibration["batches"] == row.calibration_sequences
            and calibration["tokens"] == row.calibration_sequences * 2048
            and gptq["layers"] == 40
            and gptq["linear_layers"] == 280
            and gptq["calibration_sequences"] == row.calibration_sequences
            and gptq["calibration_sequence_length"] == 2048
            and all(math.isfinite(float(value)) for value in metrics + errors)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_study_plan(output_dir: Path, rows: Tuple[StudyRow, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": "llama2-13b-wikitext2-gptq-components-and-calibration",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": [asdict(row) for row in rows],
    }
    with (output_dir / "study-plan.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _command(python: str, row: StudyRow, output_dir: Path) -> list[str]:
    config = PROJECT_ROOT / "configs" / "pipeline" / row.config
    return [python, str(PIPELINE_RUNNER), str(config), "--output", str(output_dir / "{}.json".format(row.identifier))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Ignored-artifact directory for JSON and logs")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter in the CUDA environment")
    parser.add_argument("--only", action="append", default=[], metavar="ROW", help="Run one named row; may be repeated")
    parser.add_argument("--resume", action="store_true", help="Skip only existing result JSON files with finite metrics and matching calibration count")
    parser.add_argument("--dry-run", action="store_true", help="Print the ordered commands without creating artifacts or running models")
    parser.add_argument("--validate-only", action="store_true", help="Validate every selected pipeline config without creating artifacts or loading a model")
    args = parser.parse_args()

    try:
        rows = select_rows(args.only)
    except StudyError as error:
        print("GPTQ STUDY FAILED: {}".format(error), file=sys.stderr)
        return 1

    for row in rows:
        config = PROJECT_ROOT / "configs" / "pipeline" / row.config
        if not config.is_file():
            print("GPTQ STUDY FAILED: missing config {}".format(config), file=sys.stderr)
            return 1

    if args.dry_run:
        for row in rows:
            print("{}: {}".format(row.identifier, " ".join(_command(args.python, row, args.output_dir))))
        return 0

    if args.validate_only:
        for row in rows:
            config = PROJECT_ROOT / "configs" / "pipeline" / row.config
            completed = subprocess.run([args.python, str(PIPELINE_RUNNER), str(config), "--validate-only"], cwd=PROJECT_ROOT, check=False)
            if completed.returncode != 0:
                print("GPTQ STUDY FAILED: invalid config for {}".format(row.identifier), file=sys.stderr)
                return 1
        return 0

    _write_study_plan(args.output_dir, rows)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        output = args.output_dir / "{}.json".format(row.identifier)
        if output.exists():
            if args.resume and _result_is_complete(output, row):
                print("GPTQ STUDY SKIPPED: {}".format(row.identifier))
                continue
            print("GPTQ STUDY FAILED: {} exists but is not a resumable completed result".format(output), file=sys.stderr)
            return 1
        command = _command(args.python, row, args.output_dir)
        print("GPTQ STUDY RUNNING: {}".format(row.identifier), flush=True)
        with (log_dir / "{}.log".format(row.identifier)).open("a", encoding="utf-8") as log:
            log.write("\n--- {} ---\n{}\n".format(dt.datetime.now(dt.timezone.utc).isoformat(), " ".join(command)))
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0 or not _result_is_complete(output, row):
            print("GPTQ STUDY FAILED: {} (see {})".format(row.identifier, log_dir / "{}.log".format(row.identifier)), file=sys.stderr)
            return 1
        print("GPTQ STUDY COMPLETE: {}".format(row.identifier), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
