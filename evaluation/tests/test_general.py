import json
from pathlib import Path

import pytest

from web_curation_eval.general import (
    resolve_uv_executable,
    suite_scores,
    validate_general_result,
)

SUITES = ("olmobase:easy:qa:rc",)


def _result(score: object = 0.4) -> dict:
    return {
        "tasks": [
            {
                "task": "arc_easy:rc:olmo3base",
                "metrics": {"acc:choice": 0.4},
                "primary_metric": "acc:choice",
            }
        ],
        "summary": {
            SUITES[0]: {"metric": "acc:choice", "score": score},
        },
        "errors": [],
    }


def test_complete_general_result_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_result()))

    result = validate_general_result(path, SUITES)

    assert suite_scores(result, SUITES) == {
        "olmobase:easy:qa:rc": {"metric": "acc:choice", "score": pytest.approx(0.4)}
    }


def test_partial_general_result_fails(tmp_path: Path) -> None:
    result = _result()
    result["summary"] = {}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(result))

    with pytest.raises(RuntimeError, match="missing suites"):
        validate_general_result(path, SUITES)


def test_reported_task_error_fails(tmp_path: Path) -> None:
    result = _result()
    result["errors"] = [{"task": "mmlu", "error": "boom"}]
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(result))

    with pytest.raises(RuntimeError, match="task errors"):
        validate_general_result(path, SUITES)


def test_non_numeric_general_aggregate_fails(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_result(score="NA")))

    with pytest.raises(RuntimeError, match="scores are invalid"):
        validate_general_result(path, SUITES)


def test_uv_executable_can_be_explicitly_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    monkeypatch.setenv("UV_BIN", str(uv))
    monkeypatch.setenv("PATH", "")

    assert resolve_uv_executable() == str(uv.resolve())
