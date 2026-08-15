"""Strict TOML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 evaluation environment
    import tomli as tomllib


def _required(table: dict[str, Any], key: str, kind: type) -> Any:
    if key not in table:
        raise ValueError(f"Missing required configuration key: {key}")
    value = table[key]
    if not isinstance(value, kind):
        raise TypeError(f"Configuration key {key} must be {kind.__name__}")
    return value


@dataclass(frozen=True)
class StatisticsConfig:
    dataset_repo: str
    dataset_revision: str
    split: str
    max_sequence_length: int
    batch_size: int
    dtype: str
    max_documents: int | None


@dataclass(frozen=True)
class HealthConfig:
    split: str
    target_tokens: int
    output_file: Path
    tokenizer_path: Path


@dataclass(frozen=True)
class GeneralConfig:
    framework_root: Path
    tasks: tuple[str, ...]
    max_length: int
    limit: int | None


@dataclass(frozen=True)
class EvaluationConfig:
    assets_dir: Path
    health: HealthConfig
    statistics: StatisticsConfig
    general: GeneralConfig


def load_config(path: Path) -> EvaluationConfig:
    """Load an evaluation config and reject incomplete reproducibility fields."""

    raw = tomllib.loads(path.read_text())
    assets = _required(raw, "assets", dict)
    stats = _required(raw, "statistics", dict)
    general = _required(raw, "general", dict)
    assets_dir = Path(_required(assets, "directory", str))
    revision = _required(stats, "dataset_revision", str).strip()
    if revision == "PALOMA_REVISION_REQUIRED":
        manifest_path = assets_dir / "paloma_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Paloma revision is unresolved; run prepare-assets after accepting "
                "the dataset license"
            )
        import json

        revision = str(json.loads(manifest_path.read_text())["revision"])
    if not revision or revision == "main":
        raise ValueError("statistics.dataset_revision must be an immutable revision")
    max_length = _required(stats, "max_sequence_length", int)
    if max_length < 2:
        raise ValueError("statistics.max_sequence_length must be at least 2")
    max_documents = stats.get("max_documents")
    if max_documents is not None and (
        not isinstance(max_documents, int) or max_documents < 1
    ):
        raise ValueError("statistics.max_documents must be positive or omitted")
    health = _required(raw, "health", dict)
    tasks = _required(general, "tasks", list)
    if not tasks or any(not isinstance(task, str) or not task.strip() for task in tasks):
        raise ValueError("general.tasks must contain non-empty task suite names")
    if len(tasks) != len(set(tasks)):
        raise ValueError("general.tasks must not contain duplicates")
    general_limit = general.get("limit")
    if general_limit is not None and (
        not isinstance(general_limit, int) or general_limit < 1
    ):
        raise ValueError("general.limit must be positive or omitted")
    general_max_length = _required(general, "max_length", int)
    if general_max_length < 2:
        raise ValueError("general.max_length must be at least 2")
    return EvaluationConfig(
        assets_dir=assets_dir,
        health=HealthConfig(
            split=_required(health, "split", str),
            target_tokens=_required(health, "target_tokens", int),
            output_file=Path(_required(health, "output_file", str)),
            tokenizer_path=Path(_required(health, "tokenizer_path", str)),
        ),
        statistics=StatisticsConfig(
            dataset_repo=_required(stats, "dataset_repo", str),
            dataset_revision=revision,
            split=_required(stats, "split", str),
            max_sequence_length=max_length,
            batch_size=_required(stats, "batch_size", int),
            dtype=_required(stats, "dtype", str),
            max_documents=max_documents,
        ),
        general=GeneralConfig(
            framework_root=Path(_required(general, "framework_root", str)),
            tasks=tuple(tasks),
            max_length=general_max_length,
            limit=general_limit,
        ),
    )
