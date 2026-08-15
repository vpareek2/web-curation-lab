"""Prepare immutable Paloma assets and the small training-health holdout."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 evaluation environment
    import tomli as tomllib

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, get_token, snapshot_download
from transformers import AutoTokenizer

from .constants import OLMO_EVAL_REVISION, PALOMA_REPO_ID, SCHEMA_VERSION
from .documents import infer_domain


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def general_environment(cache_path: Path) -> dict[str, str]:
    """Build the isolated general-suite environment while preserving Hub auth."""

    environment = os.environ.copy()
    environment["HF_HOME"] = str(cache_path)
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # The isolated cache intentionally freezes suite assets, but also hides the
    # user's normal Hugging Face token store. Forward the resolved token without
    # writing it to any project log or manifest.
    token = get_token()
    if token:
        environment["HF_TOKEN"] = token
    return environment


def include_for_stratified_target(
    *, current_tokens: int, document_tokens: int, target_tokens: int
) -> bool:
    """Include the document when it is the closest boundary to the target."""

    if current_tokens == 0:
        return True
    projected = current_tokens + document_tokens
    if projected < target_tokens:
        return True
    return abs(projected - target_tokens) <= abs(current_tokens - target_tokens)


def _health_holdout(
    *,
    dataset_path: Path,
    revision: str,
    split: str,
    target_tokens: int,
    tokenizer_path: Path,
    output_file: Path,
    sources: list[str],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if not sources:
        raise RuntimeError("Paloma snapshot exposes no dataset configurations")
    per_source = max(1, target_tokens // len(sources))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        total_tokens = 0
        documents = 0
        source_counts: dict[str, int] = {}
        with output_file.open("w", encoding="utf-8") as output:
            for source in sources:
                source_tokens = 0
                dataset = load_dataset(
                    str(dataset_path),
                    name=source,
                    split=split,
                    streaming=True,
                )
                for record in dataset:
                    text = record.get("text")
                    if not isinstance(text, str) or not text:
                        continue
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    if not include_for_stratified_target(
                        current_tokens=source_tokens,
                        document_tokens=len(tokens),
                        target_tokens=per_source,
                    ):
                        break
                    output.write(
                        json.dumps(
                            {
                                "source": source,
                                "domain": infer_domain(source, record),
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    source_tokens += len(tokens)
                    total_tokens += len(tokens)
                    documents += 1
                    if source_tokens >= per_source:
                        break
                source_counts[source] = source_tokens
        relative_error = abs(total_tokens - target_tokens) / target_tokens
        if relative_error <= 0.03 or attempt == 2:
            break
        per_source = max(1, round(per_source * target_tokens / total_tokens))
    return {
        "path": str(output_file),
        "sha256": sha256_file(output_file),
        "target_tokens": target_tokens,
        "tokens": total_tokens,
        "relative_error": relative_error,
        "per_source_target": per_source,
        "documents": documents,
        "tokens_by_source": source_counts,
    }


def prepare_assets(config_path: Path) -> Path:
    """Download Paloma at its resolved commit and write a hashed asset manifest."""

    raw = tomllib.loads(config_path.read_text())
    stats = raw["statistics"]
    assets_dir = Path(raw["assets"]["directory"])
    requested = stats["dataset_revision"]
    if requested == "PALOMA_REVISION_REQUIRED":
        requested = None
    repo_id = stats.get("dataset_repo", PALOMA_REPO_ID)
    try:
        info = HfApi().dataset_info(repo_id, revision=requested)
    except Exception as error:
        raise RuntimeError(
            "Unable to access Paloma. Accept the AI2 license at "
            "https://huggingface.co/datasets/allenai/paloma and authenticate "
            "with Hugging Face before retrying."
        ) from error
    revision = info.sha
    dataset_path = assets_dir / "paloma"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=dataset_path,
    )
    sources = sorted(get_dataset_config_names(str(dataset_path)))
    health = raw["health"]
    holdout = _health_holdout(
        dataset_path=dataset_path,
        revision=revision,
        split=health["split"],
        target_tokens=health["target_tokens"],
        tokenizer_path=Path(health["tokenizer_path"]),
        output_file=Path(health["output_file"]),
        sources=sources,
    )
    framework_root = Path(raw["general"]["framework_root"]).resolve()
    framework_lock = framework_root / "uv.lock"
    if not framework_lock.is_file():
        raise FileNotFoundError(f"OLMo Eval lockfile is missing: {framework_lock}")
    general_cache = assets_dir / "general_hf_cache"
    general_metadata = (assets_dir / "general_assets.json").resolve()
    script = Path(__file__).parents[2] / "scripts" / "prepare_general_assets.py"
    general_command = [
        "uv",
        "run",
        "--project",
        str(framework_root),
        "--frozen",
        "--no-group",
        "dev",
        "--no-group",
        "vllm",
        "python",
        str(script.resolve()),
        "--tasks-json",
        json.dumps(raw["general"]["tasks"]),
        "--output",
        str(general_metadata),
    ]
    general_env = general_environment(general_cache.resolve())
    try:
        subprocess.run(general_command, env=general_env, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Unable to prepare the complete OLMo Eval dataset suite; see the "
            "subprocess output above"
        ) from error
    general_details = json.loads(general_metadata.read_text())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "dataset_path": str(dataset_path),
        "sources": sources,
        "files": file_inventory(dataset_path),
        "health_holdout": holdout,
        "general": {
            "framework": "OLMo Eval",
            "revision": OLMO_EVAL_REVISION,
            "tasks": raw["general"]["tasks"],
            "framework_lock_sha256": sha256_file(framework_lock),
            "cache_path": str(general_cache),
            "cache_files": file_inventory(general_cache),
            "metadata": general_details,
            "prepare_command": general_command,
        },
    }
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = assets_dir / "paloma_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
