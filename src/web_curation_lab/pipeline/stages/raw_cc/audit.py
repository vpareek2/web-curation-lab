"""Deterministic audit samples for provisional ``0_raw_cc`` reader decisions."""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from web_curation_lab.pipeline.stages.raw_cc.probe import STAGE_ID, _load_magic
from web_curation_lab.pipeline.stages.raw_cc.reader import PayloadDecision, RawWarcReader

AUDIT_CATEGORIES = (
    "non_text_mime",
    "replacement_decode",
    "empty_text",
    "mime_disagreement",
)


@dataclass(frozen=True)
class AuditConfig:
    stage_id: str
    crawl_id: str
    policy_revision: str
    input_path: Path
    output_dir: Path
    samples_per_category: int
    preview_bytes: int
    preview_chars: int
    text_mime_prefixes: tuple[str, ...]
    text_mime_types: tuple[str, ...]
    text_mime_suffixes: tuple[str, ...]


class _BottomKSampler:
    """Keep a deterministic uniform sample using the lowest SHA-256 ranks."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, key: str, row: dict[str, Any]) -> None:
        score = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
        entry = (-score, key, row)
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, entry)
        elif score < -self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def rows(self) -> list[dict[str, Any]]:
        return [entry[2] for entry in sorted(self._heap, key=lambda item: (-item[0], item[1]))]


def load_audit_config(config_path: Path) -> AuditConfig:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        stage = raw["stage"]
        probe = raw["probe"]
        census = raw["census"]
        audit = raw["audit"]
        config = AuditConfig(
            stage_id=str(stage["id"]),
            crawl_id=str(stage["crawl_id"]),
            policy_revision=str(stage["policy_revision"]),
            input_path=Path(probe["input_path"]),
            output_dir=Path(audit["output_dir"]),
            samples_per_category=int(audit["samples_per_category"]),
            preview_bytes=int(audit["preview_bytes"]),
            preview_chars=int(audit["preview_chars"]),
            text_mime_prefixes=tuple(census["text_mime_prefixes"]),
            text_mime_types=tuple(census["text_mime_types"]),
            text_mime_suffixes=tuple(census["text_mime_suffixes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid raw-CC audit config {config_path}: {error}") from error
    if config.stage_id != STAGE_ID:
        raise ValueError(f"Expected stage.id={STAGE_ID!r}, got {config.stage_id!r}")
    if config.samples_per_category <= 0:
        raise ValueError("audit.samples_per_category must be positive")
    if config.preview_bytes < 0 or config.preview_chars < 0:
        raise ValueError("audit preview limits must be non-negative")
    if not config.policy_revision:
        raise ValueError("stage.policy_revision cannot be empty")
    return config


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mime_values(decision: PayloadDecision) -> tuple[str, str, str]:
    return (
        decision.warc_mime or "none",
        decision.http_mime or "none",
        decision.detected_mime or "none",
    )


def _mime_disagrees(decision: PayloadDecision) -> bool:
    observed = {
        value
        for value in (decision.warc_mime, decision.http_mime, decision.detected_mime)
        if value is not None
    }
    return len(observed) > 1


def _decision_categories(decision: PayloadDecision) -> list[str]:
    categories: list[str] = []
    if not decision.is_textual_mime:
        categories.append("non_text_mime")
    elif decision.decode_used_replacement:
        categories.append("replacement_decode")
    elif not decision.text:
        categories.append("empty_text")
    if _mime_disagrees(decision):
        categories.append("mime_disagreement")
    return categories


def _outcome(decision: PayloadDecision) -> str:
    if not decision.is_textual_mime:
        return "dropped_non_text_mime"
    if decision.text is None:
        return "dropped_decode_failed"
    if not decision.text:
        return "dropped_empty_text"
    if decision.decode_used_replacement:
        return "accepted_with_replacement"
    return "accepted"


def _replacement_options(payload: bytes, decision: PayloadDecision) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    tried: set[str] = set()
    for encoding in (decision.declared_charset, "utf-8", decision.detected_encoding):
        if not encoding or encoding.lower() in tried:
            continue
        tried.add(encoding.lower())
        try:
            text = payload.decode(encoding, errors="replace")
        except LookupError:
            continue
        replacements = text.count("\ufffd")
        options.append(
            {
                "encoding": encoding,
                "decoded_characters": len(text),
                "replacement_characters": replacements,
                "replacement_ratio": replacements / len(text) if text else 0.0,
            }
        )
    return options


def _replacement_ratio_bucket(ratio: float) -> str:
    if ratio <= 0.000001:
        return "le_0.0001_percent"
    if ratio <= 0.0001:
        return "le_0.01_percent"
    if ratio <= 0.001:
        return "le_0.1_percent"
    if ratio <= 0.01:
        return "le_1_percent"
    return "gt_1_percent"


def _audit_row(
    *,
    category: str,
    record_index: int,
    record: Any,
    payload: bytes,
    decision: PayloadDecision,
    preview_bytes: int,
    preview_chars: int,
) -> dict[str, Any]:
    prefix = payload[:preview_bytes]
    return {
        "audit_category": category,
        "reader_outcome": _outcome(decision),
        "record_index": record_index,
        "warc_record_id": record.rec_headers.get_header("WARC-Record-ID"),
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "warc_date": record.rec_headers.get_header("WARC-Date"),
        "http_status": record.http_headers.get_statuscode() if record.http_headers else None,
        "warc_identified_payload_type": decision.warc_mime,
        "http_content_type": decision.http_content_type,
        "http_mime_type": decision.http_mime,
        "detected_mime_type": decision.detected_mime,
        "effective_mime_type": decision.effective_mime,
        "declared_charset": decision.declared_charset,
        "detected_encoding": decision.detected_encoding,
        "encoding_confidence": decision.encoding_confidence,
        "decode_encoding": decision.decode_encoding,
        "decode_used_replacement": decision.decode_used_replacement,
        "replacement_characters": decision.replacement_characters,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_prefix_bytes": len(prefix),
        "payload_prefix_base64": base64.b64encode(prefix).decode("ascii"),
        "payload_prefix_hex": prefix.hex(),
        "payload_prefix_truncated": len(payload) > len(prefix),
        "text_preview": decision.text[:preview_chars] if decision.text is not None else None,
        "text_preview_truncated": (
            len(decision.text) > preview_chars if decision.text is not None else None
        ),
        "replacement_decode_options": (
            _replacement_options(payload, decision) if decision.decode_used_replacement else []
        ),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def run_audit(config: AuditConfig) -> dict[str, Any]:
    """Audit all response decisions and retain bounded deterministic samples."""

    from warcio.archiveiterator import ArchiveIterator

    input_path = config.input_path.resolve()
    output_dir = config.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"WARC input does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = RawWarcReader(
        str(input_path.parent),
        crawl_id=config.crawl_id,
        policy_revision=config.policy_revision,
        glob_pattern=input_path.name,
        text_mime_prefixes=config.text_mime_prefixes,
        text_mime_types=config.text_mime_types,
        text_mime_suffixes=config.text_mime_suffixes,
    )
    magic_module, magic_error = _load_magic()
    samplers = {
        category: _BottomKSampler(config.samples_per_category) for category in AUDIT_CATEGORIES
    }
    category_counts: Counter[str] = Counter()
    category_outcomes: dict[str, Counter[str]] = {
        category: Counter() for category in AUDIT_CATEGORIES
    }
    outcome_counts: Counter[str] = Counter()
    disagreement_patterns: Counter[str] = Counter()
    replacement_ratio_buckets: Counter[str] = Counter()
    best_replacement_encodings: Counter[str] = Counter()
    priority_replacement_ratio_buckets: Counter[str] = Counter()
    priority_replacement_encodings: Counter[str] = Counter()
    response_records = 0

    started = time.perf_counter()
    with input_path.open("rb") as stream:
        for record_index, record in enumerate(ArchiveIterator(stream)):
            if record.rec_type != "response":
                continue
            response_records += 1
            payload = record.content_stream().read()
            http_content_type = (
                record.http_headers.get_header("Content-Type") if record.http_headers else None
            )
            decision = reader.inspect_payload(
                payload=payload,
                warc_identified_payload_type=record.rec_headers.get_header(
                    "WARC-Identified-Payload-Type"
                ),
                http_content_type=http_content_type,
                magic_module=magic_module,
            )
            outcome_counts[_outcome(decision)] += 1
            categories = _decision_categories(decision)
            if "replacement_decode" in categories:
                replacement_options = _replacement_options(payload, decision)
                if replacement_options:
                    priority = replacement_options[0]
                    priority_replacement_ratio_buckets[
                        _replacement_ratio_bucket(float(priority["replacement_ratio"]))
                    ] += 1
                    priority_replacement_encodings[str(priority["encoding"])] += 1
                    best = min(
                        replacement_options,
                        key=lambda option: (
                            option["replacement_ratio"],
                            option["replacement_characters"],
                            option["encoding"].lower(),
                        ),
                    )
                    replacement_ratio_buckets[
                        _replacement_ratio_bucket(float(best["replacement_ratio"]))
                    ] += 1
                    best_replacement_encodings[str(best["encoding"])] += 1
            if "mime_disagreement" in categories:
                warc_mime, http_mime, detected_mime = _mime_values(decision)
                disagreement_patterns[
                    f"warc={warc_mime}|http={http_mime}|detected={detected_mime}"
                ] += 1
            record_id = record.rec_headers.get_header("WARC-Record-ID") or str(record_index)
            for category in categories:
                category_counts[category] += 1
                category_outcomes[category][_outcome(decision)] += 1
                row = _audit_row(
                    category=category,
                    record_index=record_index,
                    record=record,
                    payload=payload,
                    decision=decision,
                    preview_bytes=config.preview_bytes,
                    preview_chars=config.preview_chars,
                )
                samplers[category].add(f"{category}\0{record_id}", row)

    wall_seconds = time.perf_counter() - started
    rows = [row for category in AUDIT_CATEGORIES for row in samplers[category].rows()]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
        "policy_revision": config.policy_revision,
        "policy_status": "audit of frozen mechanical reader policy",
        "dependencies": {
            "datatrove": version("datatrove"),
            "warcio": version("warcio"),
        },
        "capabilities": {
            "libmagic_available": magic_module is not None,
            "libmagic_error": magic_error,
        },
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _file_sha256(input_path),
        },
        "sampling": {
            "method": "lowest SHA-256 ranks of category and WARC record ID",
            "samples_per_category": config.samples_per_category,
            "preview_bytes": config.preview_bytes,
            "preview_chars": config.preview_chars,
            "sample_rows_written": len(rows),
        },
        "counts": {
            "response_records": response_records,
            "reader_outcomes": dict(sorted(outcome_counts.items())),
            "audit_categories": {
                category: category_counts[category] for category in AUDIT_CATEGORIES
            },
            "audit_category_outcomes": {
                category: dict(sorted(category_outcomes[category].items()))
                for category in AUDIT_CATEGORIES
            },
            "decode_failure_replacement_audit": {
                "priority_order": "declared charset, then UTF-8, then detected encoding",
                "priority_candidate_encodings": dict(
                    sorted(priority_replacement_encodings.items())
                ),
                "priority_candidate_replacement_ratio_buckets": dict(
                    sorted(priority_replacement_ratio_buckets.items())
                ),
                "optimistic_best_candidate_note": (
                    "Lower bound only; minimizing replacements can choose the wrong encoding"
                ),
                "best_candidate_encodings": dict(sorted(best_replacement_encodings.items())),
                "best_candidate_replacement_ratio_buckets": dict(
                    sorted(replacement_ratio_buckets.items())
                ),
            },
            "mime_disagreement_patterns": dict(
                sorted(disagreement_patterns.items(), key=lambda item: (-item[1], item[0]))
            ),
        },
        "runtime": {"wall_seconds": wall_seconds},
    }
    _write_jsonl_atomic(output_dir / "records.jsonl", rows)
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary
