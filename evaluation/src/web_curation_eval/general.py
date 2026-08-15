"""Pinned OLMo Eval base-model benchmark execution and validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import EvaluationConfig
from .constants import OLMO_EVAL_REVISION, SCHEMA_VERSION
from .provenance import sha256_file, verify_inventory


def resolve_uv_executable() -> str:
    """Find uv in interactive shells, non-interactive SSH, or an override."""

    override = os.environ.get("UV_BIN")
    candidates = [override, shutil.which("uv"), str(Path.home() / ".local/bin/uv")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "Could not find the uv executable; install uv or set UV_BIN to its path"
    )


def validate_general_result(path: Path, expected_suites: tuple[str, ...]) -> dict[str, Any]:
    """Reject errored, partial, duplicate, or non-numeric suite outputs."""

    if not path.is_file():
        raise FileNotFoundError(f"OLMo Eval did not produce {path}")
    result = json.loads(path.read_text())
    errors = result.get("errors")
    if not isinstance(errors, list):
        raise RuntimeError("OLMo Eval result has no errors list")
    if errors:
        raise RuntimeError(f"OLMo Eval reported task errors: {errors}")
    tasks = result.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("OLMo Eval result has no completed tasks")
    completed_names = [task.get("task") for task in tasks if isinstance(task, dict)]
    if len(completed_names) != len(set(completed_names)):
        raise RuntimeError("OLMo Eval result contains duplicate task entries")
    summary = result.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("OLMo Eval result has no summary")
    missing = sorted(set(expected_suites) - set(summary))
    if missing:
        raise RuntimeError(f"OLMo Eval result is incomplete; missing suites: {missing}")
    invalid = [
        suite
        for suite in expected_suites
        if not isinstance(summary[suite], dict)
        or not isinstance(summary[suite].get("score"), (int, float))
        or not isinstance(summary[suite].get("metric"), str)
    ]
    if invalid:
        raise RuntimeError(f"OLMo Eval suite scores are invalid: {invalid}")
    return result


def suite_scores(result: dict[str, Any], suites: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        suite: {
            "metric": result["summary"][suite]["metric"],
            "score": float(result["summary"][suite]["score"]),
        }
        for suite in suites
    }


def run_general(
    *,
    model_path: Path,
    config: EvaluationConfig,
    output_dir: Path,
    num_gpus: int,
) -> Path:
    """Run OLMo Eval from its own frozen uv lock and preserve every raw artifact."""

    framework_root = config.general.framework_root.resolve()
    model_path = model_path.resolve()
    framework_lock = framework_root / "uv.lock"
    if not (framework_root / "src" / "olmo_eval").is_dir() or not framework_lock.is_file():
        raise FileNotFoundError(f"Pinned OLMo Eval checkout is incomplete: {framework_root}")
    manifest_path = model_path / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "General evaluation requires a packaged model with evaluation_manifest.json"
        )
    model_manifest = json.loads(manifest_path.read_text())
    verify_inventory(model_path, model_manifest["files"])
    asset_manifest_path = config.assets_dir / "paloma_manifest.json"
    if not asset_manifest_path.is_file():
        raise FileNotFoundError("Evaluation asset manifest is missing; run prepare-assets")
    asset_manifest = json.loads(asset_manifest_path.read_text())
    prepared = asset_manifest.get("general", {})
    if prepared.get("revision") != OLMO_EVAL_REVISION:
        raise ValueError("Prepared general assets use a different OLMo Eval revision")
    if tuple(prepared.get("tasks", ())) != config.general.tasks:
        raise ValueError("Prepared general suites do not match the evaluation config")
    if prepared.get("framework_lock_sha256") != sha256_file(framework_lock):
        raise ValueError("OLMo Eval lockfile does not match the prepared asset manifest")
    cache_path = Path(prepared["cache_path"]).resolve()
    verify_inventory(cache_path, prepared["cache_files"])

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    raw_dir = output_dir / "general_raw"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing general output: {raw_dir}")
    raw_dir.mkdir(exist_ok=True)

    command = [
        resolve_uv_executable(),
        "run",
        "--python",
        "3.12",
        "--project",
        str(framework_root),
        "--frozen",
        "--no-group",
        "dev",
        "olmo-eval",
        "run",
        "--harness",
        "default",
        "--override",
        "provider.kind=vllm_server",
        "--override",
        "provider.dtype=bfloat16",
        "--override",
        f"provider.max_model_len={config.general.max_length}",
        "--model",
        str(model_path),
    ]
    for task in config.general.tasks:
        command.extend(["--task", task])
        if config.general.limit is not None:
            command.extend(["--override", f"limit={config.general.limit}"])
    # This model fits on one GPU. Replicate it to evaluate independent tasks in
    # parallel instead of paying tensor-parallel communication overhead.
    command.extend(
        [
            "--num-gpus",
            "1",
            "--parallelism",
            str(num_gpus),
            "--output-dir",
            str(raw_dir.resolve()),
        ]
    )
    started = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(cache_path),
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    with (logs_dir / "general.log").open("w") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=framework_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    # Some official tasks use hf:// file URLs, whose filesystem adapter needs a
    # Hub metadata lookup even when every content blob is already cached.  Make
    # that lookup safe by rejecting any change to the prepared asset inventory.
    verify_inventory(cache_path, prepared["cache_files"])

    upstream_path = raw_dir / "metrics.json"
    result = validate_general_result(upstream_path, config.general.tasks)
    result["web_curation_provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "framework": "OLMo Eval",
        "framework_revision": OLMO_EVAL_REVISION,
        "framework_lock_sha256": sha256_file(framework_lock),
        "requested_suites": list(config.general.tasks),
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
    }
    canonical_path = output_dir / "general_raw.json"
    canonical_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    shutil.copy2(canonical_path, raw_dir / "web_curation_metrics.json")
    return canonical_path
