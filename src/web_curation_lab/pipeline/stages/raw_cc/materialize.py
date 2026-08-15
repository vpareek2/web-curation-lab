"""Resumable WARC-to-DataTrove materialization for the frozen ``0_raw_cc`` policy."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from datatrove.pipeline.tokens import DocumentTokenizer

from web_curation_lab.pipeline.stages.raw_cc.census import load_census_config
from web_curation_lab.pipeline.stages.raw_cc.probe import STAGE_ID
from web_curation_lab.pipeline.stages.raw_cc.reader import RawWarcReader
from web_curation_lab.tokenizer_assets import (
    EXPECTED_EOS_ID,
    EXPECTED_VOCAB_SIZE,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    verify_tokenizer,
)

MATERIALIZATION_SCHEMA_VERSION = 1
EOS_TOKEN_TEXT = "</s>"
READ_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class MaterializeConfig:
    stage_id: str
    crawl_id: str
    policy_revision: str
    inventory_manifest: Path
    source_dir: Path
    work_dir: Path
    output_root: Path
    tokenizer_path: Path
    tokenizer_batch_size: int
    text_mime_prefixes: tuple[str, ...]
    text_mime_types: tuple[str, ...]
    text_mime_suffixes: tuple[str, ...]
    sequence_length: int
    samples_per_shard: int
    shuffle_seed: int
    token_size_bytes: int = 2


@dataclass(frozen=True)
class InventorySource:
    index: int
    path: Path
    byte_count: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_materialize_config(config_path: Path) -> MaterializeConfig:
    """Load materialization settings while reusing the locked census policy."""

    census = load_census_config(config_path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        materialize = raw["materialize"]
        config = MaterializeConfig(
            stage_id=census.stage_id,
            crawl_id=census.crawl_id,
            policy_revision=census.policy_revision,
            inventory_manifest=Path(materialize["inventory_manifest"]),
            source_dir=Path(materialize["source_dir"]),
            work_dir=Path(materialize["work_dir"]),
            output_root=Path(materialize["output_root"]),
            tokenizer_path=census.tokenizer_path,
            tokenizer_batch_size=census.tokenizer_batch_size,
            text_mime_prefixes=census.text_mime_prefixes,
            text_mime_types=census.text_mime_types,
            text_mime_suffixes=census.text_mime_suffixes,
            sequence_length=int(materialize["sequence_length"]),
            samples_per_shard=int(materialize["samples_per_shard"]),
            shuffle_seed=int(materialize["shuffle_seed"]),
            token_size_bytes=int(materialize.get("token_size_bytes", 2)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid raw-CC materialize config {config_path}: {error}") from error
    if config.stage_id != STAGE_ID:
        raise ValueError(f"Expected stage.id={STAGE_ID!r}, got {config.stage_id!r}")
    if config.sequence_length <= 0:
        raise ValueError("materialize.sequence_length must be positive")
    if config.samples_per_shard <= 0:
        raise ValueError("materialize.samples_per_shard must be positive")
    if config.token_size_bytes != 2:
        raise ValueError("The pinned 32K tokenizer requires two-byte token storage")
    return config


def _load_inventory(
    config: MaterializeConfig, limit: int | None
) -> tuple[list[InventorySource], str]:
    inventory_path = config.inventory_manifest.resolve()
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Materialization inventory does not exist: {inventory_path}")
    try:
        inventory = json.loads(inventory_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"Could not read materialization inventory {inventory_path}: {error}"
        ) from error
    expected_identity = (
        MATERIALIZATION_SCHEMA_VERSION,
        config.stage_id,
        config.crawl_id,
        config.policy_revision,
    )
    actual_identity = (
        inventory.get("schema_version"),
        inventory.get("stage_id"),
        inventory.get("crawl_id"),
        inventory.get("policy_revision"),
    )
    if actual_identity != expected_identity:
        raise ValueError(
            "Inventory identity does not match the materialization config: "
            f"expected {expected_identity}, found {actual_identity}"
        )
    rows = inventory.get("downloads")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Materialization inventory contains no downloads")
    if limit is not None:
        if limit <= 0:
            raise ValueError("Materialization limit must be positive")
        if limit > len(rows):
            raise ValueError(
                f"Materialization limit {limit} exceeds inventory size {len(rows)}"
            )
        rows = rows[:limit]

    source_dir = config.source_dir.resolve()
    sources: list[InventorySource] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for index, row in enumerate(rows):
        try:
            filename = Path(str(row["local_path"])).name
            expected_bytes = int(row["bytes"])
            expected_hash = str(row["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid inventory row {index}: {error}") from error
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Inventory WARC does not exist: {path}")
        if path in seen_paths or expected_hash in seen_hashes:
            raise ValueError(f"Inventory contains a duplicate WARC at row {index}: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Inventory source size mismatch for {path}: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Inventory source SHA-256 mismatch for {path}")
        seen_paths.add(path)
        seen_hashes.add(expected_hash)
        sources.append(InventorySource(index, path, expected_bytes, expected_hash))
    return sources, _sha256(inventory_path)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_file_record(folder: Path, record: dict[str, Any]) -> Path:
    path = folder / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Materialization artifact does not exist: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != int(record["bytes"]):
        raise ValueError(
            f"Materialization artifact size mismatch for {path}: "
            f"expected {record['bytes']}, found {actual_bytes}"
        )
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Materialization artifact SHA-256 mismatch for {path}")
    return path


def _source_identity(config: MaterializeConfig, source: InventorySource) -> dict[str, Any]:
    return {
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
        "policy_revision": config.policy_revision,
        "source_index": source.index,
        "source_sha256": source.sha256,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files_sha256": TOKENIZER_SHA256,
        "textual_mime_policy": {
            "prefixes": list(config.text_mime_prefixes),
            "types": list(config.text_mime_types),
            "suffixes": list(config.text_mime_suffixes),
        },
        "dependencies": {
            "datatrove": version("datatrove"),
            "tokenizers": version("tokenizers"),
            "warcio": version("warcio"),
        },
        "shuffle_seed": config.shuffle_seed + source.index,
        "token_size_bytes": config.token_size_bytes,
    }


def _load_completed_source(
    folder: Path,
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = folder / "source_manifest.json"
    if not folder.exists():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"Completed source directory has no manifest: {folder}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"Could not read completed source manifest {manifest_path}: {error}"
        ) from error
    if manifest.get("identity") != expected_identity:
        raise ValueError(f"Completed source identity mismatch: {manifest_path}")
    _validate_file_record(folder, manifest["artifacts"]["tokens"])
    _validate_file_record(folder, manifest["artifacts"]["index"])
    return manifest


def _tokenize_source(config: MaterializeConfig, source: InventorySource) -> dict[str, Any]:
    work_dir = config.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    key = f"{source.index:05d}-{source.sha256[:12]}"
    final_dir = work_dir / key
    identity = _source_identity(config, source)
    completed = _load_completed_source(final_dir, expected_identity=identity)
    if completed is not None:
        return completed

    temporary_dir = work_dir / f".{key}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()
    reader = RawWarcReader(
        str(source.path.parent),
        crawl_id=config.crawl_id,
        policy_revision=config.policy_revision,
        glob_pattern=source.path.name,
        text_mime_prefixes=config.text_mime_prefixes,
        text_mime_types=config.text_mime_types,
        text_mime_suffixes=config.text_mime_suffixes,
    )
    tokenizer = DocumentTokenizer(
        output_folder=str(temporary_dir),
        tokenizer_name_or_path=str(config.tokenizer_path.resolve()),
        save_filename=key,
        eos_token=EOS_TOKEN_TEXT,
        save_index=True,
        save_loss_metadata=False,
        save_final_metadata=True,
        batch_size=config.tokenizer_batch_size,
        seed=config.shuffle_seed + source.index,
        shuffle_documents=True,
    )
    tokenizer(reader(), rank=0, world_size=1)

    token_paths = sorted(temporary_dir.glob("*.ds"))
    if len(token_paths) != 1:
        raise ValueError(
            f"Expected exactly one token file for {source.path}, found {len(token_paths)}"
        )
    token_path = token_paths[0]
    index_path = token_path.with_suffix(token_path.suffix + ".index")
    if not index_path.is_file():
        raise FileNotFoundError(f"DataTrove did not write an index for {token_path}")
    if token_path.stat().st_size % config.token_size_bytes:
        raise ValueError(f"Token file has an invalid byte size: {token_path}")
    if index_path.stat().st_size % 8:
        raise ValueError(f"Token index has an invalid byte size: {index_path}")
    token_count = token_path.stat().st_size // config.token_size_bytes
    document_count = index_path.stat().st_size // 8
    if token_count == 0 or document_count == 0:
        raise ValueError(f"Materialized source is empty: {source.path}")
    with token_path.open("rb") as handle:
        handle.seek(-config.token_size_bytes, os.SEEK_END)
        final_token = int.from_bytes(handle.read(config.token_size_bytes), "little")
    if final_token != EXPECTED_EOS_ID:
        raise ValueError(
            f"Materialized source does not end with EOS {EXPECTED_EOS_ID}: {token_path}"
        )
    doc_ends = np.frombuffer(index_path.read_bytes(), dtype="<u8")
    if not len(doc_ends) or int(doc_ends[-1]) != token_count:
        raise ValueError(f"Final document boundary does not match token count: {index_path}")

    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "identity": identity,
        "input": {
            "filename": source.path.name,
            "bytes": source.byte_count,
            "sha256": source.sha256,
        },
        "counts": {
            "documents": document_count,
            "tokens_with_eos": token_count,
        },
        "artifacts": {
            "tokens": _file_record(token_path),
            "index": _file_record(index_path),
        },
    }
    _write_json_atomic(temporary_dir / "source_manifest.json", manifest)
    temporary_dir.replace(final_dir)
    return manifest


def assemble_token_streams(
    *,
    input_paths: list[Path],
    output_dir: Path,
    sequence_length: int,
    samples_per_shard: int,
    eos_token_id: int,
    token_size_bytes: int,
) -> dict[str, Any]:
    """Concatenate token streams into sample-aligned shards without loading them whole."""

    if not input_paths:
        raise ValueError("No token streams were provided for assembly")
    if sequence_length <= 0 or samples_per_shard <= 0:
        raise ValueError("Sequence length and samples per shard must be positive")
    if token_size_bytes not in (2, 4):
        raise ValueError("token_size_bytes must be two or four")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Assembly output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_sample = sequence_length + 1
    target_tokens_per_shard = samples_per_shard * tokens_per_sample
    dtype = np.dtype("<u2" if token_size_bytes == 2 else "<u4")
    shard_records: list[dict[str, Any]] = []
    shard_handle = None
    shard_path: Path | None = None
    shard_tokens = 0
    shard_boundaries: list[int] = []
    source_tokens = 0

    def open_shard() -> None:
        nonlocal shard_handle, shard_path, shard_tokens, shard_boundaries
        shard_path = output_dir / f"{len(shard_records):05d}.ds"
        shard_handle = shard_path.open("wb")
        shard_tokens = 0
        shard_boundaries = []

    def close_shard(*, keep_tokens: int | None = None) -> None:
        nonlocal shard_handle, shard_path, shard_tokens, shard_boundaries
        if shard_handle is None or shard_path is None:
            return
        if keep_tokens is not None and keep_tokens < shard_tokens:
            shard_handle.truncate(keep_tokens * token_size_bytes)
            shard_tokens = keep_tokens
            shard_boundaries = [end for end in shard_boundaries if end <= keep_tokens]
        shard_handle.close()
        shard_handle = None
        if shard_tokens == 0:
            shard_path.unlink()
            shard_path = None
            return
        index_path = shard_path.with_suffix(shard_path.suffix + ".index")
        index_path.write_bytes(np.asarray(shard_boundaries, dtype="<u8").tobytes())
        shard_records.append(
            {
                "path": shard_path.name,
                "bytes": shard_path.stat().st_size,
                "sha256": _sha256(shard_path),
                "index_path": index_path.name,
                "index_bytes": index_path.stat().st_size,
                "index_sha256": _sha256(index_path),
                "tokens": shard_tokens,
                "samples": shard_tokens // tokens_per_sample,
                "document_boundaries": len(shard_boundaries),
            }
        )
        shard_path = None

    open_shard()
    for input_path in input_paths:
        byte_count = input_path.stat().st_size
        if byte_count % token_size_bytes:
            raise ValueError(f"Input token file has an invalid byte size: {input_path}")
        source_tokens += byte_count // token_size_bytes
        with input_path.open("rb") as source_handle:
            while True:
                remaining_tokens = target_tokens_per_shard - shard_tokens
                read_bytes = min(READ_CHUNK_BYTES, remaining_tokens * token_size_bytes)
                chunk = source_handle.read(read_bytes)
                if not chunk:
                    break
                if len(chunk) % token_size_bytes:
                    raise ValueError(f"Short token read from {input_path}")
                assert shard_handle is not None
                shard_handle.write(chunk)
                chunk_tokens = np.frombuffer(chunk, dtype=dtype)
                eos_offsets = np.flatnonzero(chunk_tokens == eos_token_id)
                shard_boundaries.extend(
                    (eos_offsets + shard_tokens + 1).astype(int).tolist()
                )
                shard_tokens += len(chunk_tokens)
                if shard_tokens == target_tokens_per_shard:
                    close_shard()
                    open_shard()

    full_tail_tokens = (shard_tokens // tokens_per_sample) * tokens_per_sample
    discarded_final_tokens = shard_tokens - full_tail_tokens
    close_shard(keep_tokens=full_tail_tokens)
    materialized_tokens = sum(int(record["tokens"]) for record in shard_records)
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "sequence_length": sequence_length,
        "tokens_per_sample": tokens_per_sample,
        "samples_per_shard": samples_per_shard,
        "token_size_bytes": token_size_bytes,
        "counts": {
            "source_tokens": source_tokens,
            "materialized_tokens": materialized_tokens,
            "discarded_final_tokens": discarded_final_tokens,
            "samples": materialized_tokens // tokens_per_sample,
            "document_boundaries": sum(
                int(record["document_boundaries"]) for record in shard_records
            ),
            "shards": len(shard_records),
        },
        "shards": shard_records,
    }


def _assembly_identity(
    config: MaterializeConfig,
    sources: list[InventorySource],
    source_manifests: list[dict[str, Any]],
) -> tuple[str, list[int]]:
    ordering = np.random.default_rng(config.shuffle_seed).permutation(len(sources)).tolist()
    value = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "policy_revision": config.policy_revision,
        "tokenizer_revision": TOKENIZER_REVISION,
        "sequence_length": config.sequence_length,
        "samples_per_shard": config.samples_per_shard,
        "shuffle_seed": config.shuffle_seed,
        "ordered_source_artifacts": [
            source_manifests[index]["artifacts"]["tokens"]["sha256"] for index in ordering
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), [int(index) for index in ordering]


def _load_completed_assembly(folder: Path, fingerprint: str) -> dict[str, Any] | None:
    manifest_path = folder / "materialization_manifest.json"
    if not folder.exists():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"Completed assembly directory has no manifest: {folder}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read assembly manifest {manifest_path}: {error}") from error
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(f"Assembly fingerprint mismatch: {manifest_path}")
    for shard in manifest["shards"]:
        _validate_file_record(folder, shard)
        _validate_file_record(
            folder,
            {
                "path": shard["index_path"],
                "bytes": shard["index_bytes"],
                "sha256": shard["index_sha256"],
            },
        )
    return manifest


def run_materialization(
    config: MaterializeConfig,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Materialize an integrity-checked inventory prefix into training-ready shards."""

    # Keep CLI stdout machine-readable; the shared verifier reports its human
    # confirmation with ``print``.
    with contextlib.redirect_stdout(sys.stderr):
        verify_tokenizer(config.tokenizer_path.resolve().parent)
    sources, inventory_hash = _load_inventory(config, limit)
    source_manifests = [_tokenize_source(config, source) for source in sources]
    fingerprint, ordering = _assembly_identity(config, sources, source_manifests)
    final_dir = config.output_root.resolve() / f"materialized-{fingerprint[:12]}"
    completed = _load_completed_assembly(final_dir, fingerprint)
    if completed is not None:
        return completed

    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".materialized-{fingerprint[:12]}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()
    input_paths = [
        config.work_dir.resolve()
        / f"{sources[index].index:05d}-{sources[index].sha256[:12]}"
        / source_manifests[index]["artifacts"]["tokens"]["path"]
        for index in ordering
    ]
    assembly = assemble_token_streams(
        input_paths=input_paths,
        output_dir=temporary_dir,
        sequence_length=config.sequence_length,
        samples_per_shard=config.samples_per_shard,
        eos_token_id=EXPECTED_EOS_ID,
        token_size_bytes=config.token_size_bytes,
    )
    if assembly["counts"]["shards"] == 0:
        raise ValueError("Materialization did not produce one complete training sample")
    source_documents = sum(
        int(manifest["counts"]["documents"]) for manifest in source_manifests
    )
    manifest = {
        **assembly,
        "fingerprint": fingerprint,
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
        "policy_revision": config.policy_revision,
        "dataset_dir": str(final_dir),
        "inventory": {
            "path": str(config.inventory_manifest.resolve()),
            "sha256": inventory_hash,
            "selected_prefix": len(sources),
        },
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "files_sha256": TOKENIZER_SHA256,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "eos_id": EXPECTED_EOS_ID,
        },
        "shuffle": {
            "seed": config.shuffle_seed,
            "source_order": ordering,
            "strategy": "deterministic documents within WARC, then deterministic WARC order",
        },
        "dependencies": {
            "datatrove": version("datatrove"),
            "tokenizers": version("tokenizers"),
            "warcio": version("warcio"),
        },
        "counts": {
            **assembly["counts"],
            "source_warcs": len(sources),
            "source_documents": source_documents,
        },
        "sources": [
            {
                "source_index": source.index,
                "filename": source.path.name,
                "bytes": source.byte_count,
                "sha256": source.sha256,
                "documents": source_manifests[index]["counts"]["documents"],
                "tokens_with_eos": source_manifests[index]["counts"]["tokens_with_eos"],
                "token_artifact_sha256": source_manifests[index]["artifacts"]["tokens"][
                    "sha256"
                ],
            }
            for index, source in enumerate(sources)
        ],
    }
    _write_json_atomic(temporary_dir / "materialization_manifest.json", manifest)
    temporary_dir.replace(final_dir)
    return manifest
