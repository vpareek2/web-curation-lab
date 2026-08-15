"""Artifact integrity and installed-environment provenance."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inventory(root: Path, files: list[dict[str, Any]]) -> None:
    for entry in files:
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Manifest size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Manifest SHA-256 mismatch: {path}")


def installed_versions(names: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result
