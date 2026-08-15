"""Resumable multi-WARC yield pilot for the frozen ``0_raw_cc`` policy."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import statistics
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from web_curation_lab.pipeline.stages.raw_cc.census import (
    CensusConfig,
    load_census_config,
    run_census,
)
from web_curation_lab.pipeline.stages.raw_cc.probe import STAGE_ID
from web_curation_lab.tokenizer_assets import (
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_VOCAB_SIZE,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)

SELECTION_ALGORITHM = "rounded evenly spaced zero-based indices including both endpoints"


@dataclass(frozen=True)
class PilotConfig:
    stage_id: str
    crawl_id: str
    policy_revision: str
    listing_path: Path
    source_dir: Path
    output_dir: Path
    base_url: str
    sample_size: int
    expected_listing_entries: int
    target_training_tokens: int
    download_chunk_bytes: int
    download_timeout_seconds: int
    download_retries: int


def load_pilot_config(config_path: Path) -> tuple[PilotConfig, CensusConfig]:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        stage = raw["stage"]
        pilot = raw["pilot"]
        config = PilotConfig(
            stage_id=str(stage["id"]),
            crawl_id=str(stage["crawl_id"]),
            policy_revision=str(stage["policy_revision"]),
            listing_path=Path(pilot["listing_path"]),
            source_dir=Path(pilot["source_dir"]),
            output_dir=Path(pilot["output_dir"]),
            base_url=str(pilot["base_url"]).rstrip("/"),
            sample_size=int(pilot["sample_size"]),
            expected_listing_entries=int(pilot["expected_listing_entries"]),
            target_training_tokens=int(pilot["target_training_tokens"]),
            download_chunk_bytes=int(pilot["download_chunk_bytes"]),
            download_timeout_seconds=int(pilot["download_timeout_seconds"]),
            download_retries=int(pilot["download_retries"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid raw-CC pilot config {config_path}: {error}") from error
    if config.stage_id != STAGE_ID:
        raise ValueError(f"Expected stage.id={STAGE_ID!r}, got {config.stage_id!r}")
    if config.sample_size < 2:
        raise ValueError("pilot.sample_size must be at least two")
    if config.expected_listing_entries < config.sample_size:
        raise ValueError("pilot.expected_listing_entries must cover the sample")
    if config.target_training_tokens <= 0:
        raise ValueError("pilot.target_training_tokens must be positive")
    if config.download_chunk_bytes <= 0 or config.download_timeout_seconds <= 0:
        raise ValueError("pilot download chunk and timeout values must be positive")
    if config.download_retries <= 0:
        raise ValueError("pilot.download_retries must be positive")

    census = load_census_config(config_path)
    if (census.stage_id, census.crawl_id, census.policy_revision) != (
        config.stage_id,
        config.crawl_id,
        config.policy_revision,
    ):
        raise ValueError("Pilot and census policy identities do not match")
    return config, census


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Return deterministic nearest-integer indices, including both endpoints."""

    if count < 2 or total < count:
        raise ValueError("Need total >= count >= 2")
    denominator = count - 1
    return [
        (index * (total - 1) + denominator // 2) // denominator for index in range(count)
    ]


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


def _read_listing(config: PilotConfig) -> list[str]:
    listing_path = config.listing_path.resolve()
    if not listing_path.is_file():
        raise FileNotFoundError(f"WARC listing does not exist: {listing_path}")
    with gzip.open(listing_path, "rt") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    if len(paths) != config.expected_listing_entries:
        raise ValueError(
            f"Expected {config.expected_listing_entries} WARC paths, found {len(paths)}"
        )
    prefix = f"crawl-data/{config.crawl_id}/"
    invalid = [
        path
        for path in paths
        if not path.startswith(prefix) or not path.endswith(".warc.gz")
    ]
    if invalid:
        raise ValueError(f"Listing contains an invalid path for {config.crawl_id}: {invalid[0]}")
    return paths


def _selection_manifest(config: PilotConfig, paths: list[str]) -> dict[str, Any]:
    indices = evenly_spaced_indices(len(paths), config.sample_size)
    source_dir = config.source_dir.resolve()
    selected = []
    for index in indices:
        common_crawl_path = paths[index]
        selected.append(
            {
                "selection_index_zero_based": index,
                "listing_line_one_based": index + 1,
                "common_crawl_path": common_crawl_path,
                "url": f"{config.base_url}/{common_crawl_path}",
                "local_path": str(source_dir / Path(common_crawl_path).name),
            }
        )
    return {
        "schema_version": 1,
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
        "policy_revision": config.policy_revision,
        "selection_algorithm": SELECTION_ALGORITHM,
        "listing": {
            "path": str(config.listing_path.resolve()),
            "entries": len(paths),
            "bytes": config.listing_path.resolve().stat().st_size,
            "sha256": _sha256(config.listing_path.resolve()),
        },
        "selected": selected,
    }


def _validate_gzip(path: Path, chunk_bytes: int) -> None:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(chunk_bytes):
                pass
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ValueError(f"Invalid or incomplete gzip file {path}: {error}") from error


def _download_once(
    *,
    url: str,
    destination: Path,
    chunk_bytes: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    partial = destination.with_suffix(destination.suffix + ".part")
    existing_bytes = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "web-curation-lab/0_raw_cc-pilot"}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        if error.code == 416 and existing_bytes:
            content_range = error.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes \*/(\d+)", content_range)
            if match and int(match.group(1)) == existing_bytes:
                partial.replace(destination)
                return {"resumed_from_bytes": existing_bytes, "http_status": 416}
        raise

    with response:
        status = getattr(response, "status", response.getcode())
        mode = "ab" if existing_bytes and status == 206 else "wb"
        if mode == "ab":
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing_bytes}-"):
                raise RuntimeError(
                    f"Unexpected Content-Range while resuming {url}: {content_range!r}"
                )
        resumed_from = existing_bytes if mode == "ab" else 0
        downloaded = resumed_from
        last_report = time.monotonic()
        with partial.open(mode) as handle:
            while chunk := response.read(chunk_bytes):
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 10:
                    print(f"download {destination.name}: {downloaded / 1e9:.3f} GB", flush=True)
                    last_report = now
    partial.replace(destination)
    return {"resumed_from_bytes": resumed_from, "http_status": status}


def _download(
    *,
    url: str,
    destination: Path,
    config: PilotConfig,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        transfer = {"status": "reused_existing", "resumed_from_bytes": None, "http_status": None}
    else:
        for attempt in range(1, config.download_retries + 1):
            try:
                transfer = {
                    "status": "downloaded",
                    **_download_once(
                        url=url,
                        destination=destination,
                        chunk_bytes=config.download_chunk_bytes,
                        timeout_seconds=config.download_timeout_seconds,
                    ),
                }
                break
            except (OSError, urllib.error.URLError) as error:
                if attempt == config.download_retries:
                    raise RuntimeError(
                        f"Download failed after {attempt} attempts for {url}: {error}"
                    ) from error
                delay = min(2 ** (attempt - 1), 16)
                print(f"download retry {attempt}/{config.download_retries}: {error}", flush=True)
                time.sleep(delay)
    _validate_gzip(destination, config.download_chunk_bytes)
    result = {
        **transfer,
        "url": url,
        "local_path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "gzip_valid": True,
    }
    print(
        f"ready {destination.name}: {result['bytes'] / 1e9:.3f} GB "
        f"sha256={result['sha256'][:12]}…",
        flush=True,
    )
    return result


def _matching_census(summary_path: Path, download: dict[str, Any], config: PilotConfig) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        summary.get("policy_revision") == config.policy_revision
        and summary.get("input", {}).get("sha256") == download["sha256"]
    )


def _census_one(
    *,
    entry: dict[str, Any],
    download: dict[str, Any],
    config: PilotConfig,
    census_config: CensusConfig,
) -> tuple[dict[str, Any], Path, str]:
    input_path = Path(download["local_path"])
    key = input_path.name.removesuffix(".warc.gz")
    output_dir = config.output_dir.resolve() / "census" / key
    summary_path = output_dir / "summary.json"
    legacy_summary = census_config.output_dir.resolve() / "summary.json"
    for candidate in (summary_path, legacy_summary):
        if _matching_census(candidate, download, config):
            print(f"reuse census {input_path.name}: {candidate}", flush=True)
            return json.loads(candidate.read_text()), candidate, "reused"

    print(
        f"census line {entry['listing_line_one_based']}/{config.expected_listing_entries}: "
        f"{input_path.name}",
        flush=True,
    )
    summary = run_census(
        replace(census_config, input_path=input_path, output_dir=output_dir)
    )
    return summary, summary_path, "completed"


def _distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_stddev": statistics.pstdev(values),
        "coefficient_of_variation": (
            statistics.pstdev(values) / statistics.fmean(values)
            if statistics.fmean(values)
            else 0.0
        ),
    }


def _combined_summary(
    config: PilotConfig,
    selection: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    token_values = [run["counts"]["training_tokens_with_eos"] for run in runs]
    document_values = [run["counts"]["decoded_documents"] for run in runs]
    compressed_values = [run["input"]["bytes"] for run in runs]
    mean_tokens = statistics.fmean(token_values)
    return {
        "schema_version": 1,
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
        "policy_revision": config.policy_revision,
        "selection_algorithm": selection["selection_algorithm"],
        "sample_size": len(runs),
        "target_training_tokens": config.target_training_tokens,
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "bos_id": EXPECTED_BOS_ID,
            "eos_id": EXPECTED_EOS_ID,
            "file_sha256": TOKENIZER_SHA256,
        },
        "distributions": {
            "training_tokens_with_eos_per_warc": _distribution(token_values),
            "decoded_documents_per_warc": _distribution(document_values),
            "compressed_bytes_per_warc": _distribution(compressed_values),
        },
        "sample_totals": {
            "training_tokens_with_eos": sum(token_values),
            "decoded_documents": sum(document_values),
            "compressed_bytes": sum(compressed_values),
        },
        "estimate": {
            "warc_files_for_target_at_sample_mean": math.ceil(
                config.target_training_tokens / mean_tokens
            ),
            "note": "Pilot estimate only; the final inventory requires measured cumulative totals",
        },
        "warcs": [
            {
                "listing_line_one_based": selection["selected"][index][
                    "listing_line_one_based"
                ],
                "common_crawl_path": selection["selected"][index]["common_crawl_path"],
                "input": run["input"],
                "census_summary": run["_summary_path"],
                "counts": {
                    "response_records": run["counts"]["response_records"],
                    "decoded_documents": run["counts"]["decoded_documents"],
                    "decoded_with_replacement": run["counts"]["decoded_with_replacement"],
                    "dropped_non_text_mime": run["counts"]["dropped_non_text_mime"],
                    "dropped_empty_text": run["counts"]["dropped_empty_text"],
                    "training_tokens_with_eos": run["counts"]["training_tokens_with_eos"],
                },
            }
            for index, run in enumerate(runs)
        ],
    }


def run_pilot(config: PilotConfig, census_config: CensusConfig) -> dict[str, Any]:
    """Select, download, census, and summarize the deterministic WARC pilot."""

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _read_listing(config)
    selection = _selection_manifest(config, paths)
    _write_json_atomic(output_dir / "selection_manifest.json", selection)

    download_rows: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(selection["selected"], start=1):
        print(f"download {ordinal}/{config.sample_size}: {entry['url']}", flush=True)
        download_rows.append(
            _download(
                url=entry["url"],
                destination=Path(entry["local_path"]),
                config=config,
            )
        )
        _write_json_atomic(
            output_dir / "downloads_manifest.json",
            {
                "schema_version": 1,
                "stage_id": config.stage_id,
                "crawl_id": config.crawl_id,
                "policy_revision": config.policy_revision,
                "downloads": download_rows,
            },
        )

    census_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for ordinal, (entry, download) in enumerate(
        zip(selection["selected"], download_rows, strict=True), start=1
    ):
        print(
            f"process {ordinal}/{config.sample_size}: {Path(download['local_path']).name}",
            flush=True,
        )
        summary, summary_path, status = _census_one(
            entry=entry,
            download=download,
            config=config,
            census_config=census_config,
        )
        summary["_summary_path"] = str(summary_path)
        run_summaries.append(summary)
        census_rows.append(
            {
                "listing_line_one_based": entry["listing_line_one_based"],
                "input_sha256": download["sha256"],
                "summary_path": str(summary_path),
                "status": status,
                "training_tokens_with_eos": summary["counts"]["training_tokens_with_eos"],
            }
        )
        _write_json_atomic(
            output_dir / "census_manifest.json",
            {
                "schema_version": 1,
                "stage_id": config.stage_id,
                "crawl_id": config.crawl_id,
                "policy_revision": config.policy_revision,
                "censuses": census_rows,
            },
        )

    combined = _combined_summary(config, selection, run_summaries)
    _write_json_atomic(output_dir / "summary.json", combined)
    return combined
