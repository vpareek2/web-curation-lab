"""Run both first-class evaluation suites and write a compact summary."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .config import EvaluationConfig
from .general import run_general, suite_scores


def run_all(
    *,
    model_path: Path,
    config_path: Path,
    config: EvaluationConfig,
    output_dir: Path,
    num_gpus: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    statistics_path = output_dir / "statistics.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={num_gpus}",
        "-m",
        "web_curation_eval.cli",
        "statistics",
        "--model",
        str(model_path.resolve()),
        "--config",
        str(config_path.resolve()),
        "--output",
        str(statistics_path.resolve()),
    ]
    started = time.monotonic()
    with (logs_dir / "statistics.log").open("w") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    general_path = run_general(
        model_path=model_path,
        config=config,
        output_dir=output_dir,
        num_gpus=num_gpus,
    )
    statistics = json.loads(statistics_path.read_text())
    general = json.loads(general_path.read_text())
    model_manifest = json.loads((model_path / "evaluation_manifest.json").read_text())
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": 1,
        "model": model_manifest,
        "statistics": {
            "micro": statistics["micro"],
            "macro": statistics["macro"],
            "domains": len(statistics["domains"]),
        },
        "general": {
            "framework": "OLMo Eval",
            "suite_scores": suite_scores(general, config.general.tasks),
        },
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": {
            "statistics": "statistics.json",
            "general": "general_raw.json",
            "general_raw_directory": "general_raw",
            "logs": "logs",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path
