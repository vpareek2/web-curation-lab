"""Bounded inspection of raw Common Crawl WARC records.

This probe deliberately reads records below DataTrove's standard ``WarcReader``
so we can measure what that reader would discard before defining the
``0_raw_cc`` payload policy.
"""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, replace
from email.message import Message
from importlib.metadata import version
from pathlib import Path
from typing import Any

STAGE_ID = "0_raw_cc"
DATATROVE_HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})


@dataclass(frozen=True)
class ProbeConfig:
    """Validated inputs for one bounded raw-WARC inspection."""

    stage_id: str
    crawl_id: str
    input_path: Path
    output_dir: Path
    limit: int
    preview_chars: int


def load_probe_config(config_path: Path, limit_override: int | None = None) -> ProbeConfig:
    """Load and validate a ``0_raw_cc`` TOML probe configuration."""

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        stage = raw["stage"]
        probe = raw["probe"]
        config = ProbeConfig(
            stage_id=str(stage["id"]),
            crawl_id=str(stage["crawl_id"]),
            input_path=Path(probe["input_path"]),
            output_dir=Path(probe["output_dir"]),
            limit=int(probe["limit"]),
            preview_chars=int(probe["preview_chars"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid raw-CC probe config {config_path}: {error}") from error
    if limit_override is not None:
        config = replace(config, limit=limit_override)
    if config.stage_id != STAGE_ID:
        raise ValueError(f"Expected stage.id={STAGE_ID!r}, got {config.stage_id!r}")
    if not config.crawl_id.startswith("CC-MAIN-"):
        raise ValueError(f"Invalid Common Crawl identifier: {config.crawl_id!r}")
    if config.limit <= 0:
        raise ValueError("probe.limit must be positive")
    if config.preview_chars < 0:
        raise ValueError("probe.preview_chars must be non-negative")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_mime(value: str | None) -> str | None:
    if value is None:
        return None
    mime = value.partition(";")[0].strip().lower()
    return mime or None


def _http_content_type(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    message = Message()
    message["content-type"] = value
    return _normalized_mime(message.get_content_type()), message.get_content_charset()


def _load_magic() -> tuple[Any | None, str | None]:
    try:
        import magic
    except (ImportError, OSError) as error:
        return None, str(error)
    return magic, None


def _detect_payload(
    payload: bytes, magic_module: Any | None
) -> tuple[str | None, str | None, float | None]:
    import cchardet

    detected_mime = (
        _normalized_mime(magic_module.from_buffer(payload, mime=True))
        if payload and magic_module is not None
        else None
    )
    detected_encoding: str | None = None
    confidence: float | None = None
    if payload:
        detection = cchardet.detect(payload)
        detected_encoding = detection.get("encoding")
        raw_confidence = detection.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else None
    return detected_mime, detected_encoding, confidence


def _decode_preview(
    payload: bytes,
    declared_charset: str | None,
    detected_encoding: str | None,
    preview_chars: int,
) -> tuple[str, str, bool]:
    candidates = [declared_charset, "utf-8", detected_encoding]
    tried: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return payload.decode(candidate)[:preview_chars], candidate, True
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")[:preview_chars], "utf-8-replace", False


def _datatrove_standard_decision(
    *,
    record_type: str | None,
    warc_mime: str | None,
    detected_mime: str | None,
    payload: bytes,
) -> tuple[bool, str]:
    """Mirror DataTrove 0.9.0 ``WarcReader.process_record`` decisions."""

    if record_type not in {"response", "conversion"}:
        return False, "record_type"
    effective_mime = warc_mime if warc_mime is not None else detected_mime
    allowed = effective_mime in DATATROVE_HTML_MIME_TYPES or (
        record_type == "conversion" and effective_mime == "text/plain"
    )
    if not allowed:
        return False, "mime_type"
    try:
        payload.decode("UTF-8")
        return True, "kept"
    except UnicodeDecodeError:
        import cchardet

        encoding = cchardet.detect(payload).get("encoding")
        if not encoding or encoding == "UTF-8":
            return False, "decode_failed"
        try:
            payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            return False, "decode_failed"
    return True, "kept"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


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


def run_probe(config: ProbeConfig) -> dict[str, Any]:
    """Inspect a bounded prefix of one WARC and write records plus a summary."""

    from warcio.archiveiterator import ArchiveIterator

    input_path = config.input_path.resolve()
    output_dir = config.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"WARC input does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    record_types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    warc_mimes: Counter[str] = Counter()
    http_mimes: Counter[str] = Counter()
    detected_mimes: Counter[str] = Counter()
    datatrove_decisions: Counter[str] = Counter()
    total_payload_bytes = 0
    magic_module, magic_error = _load_magic()

    with input_path.open("rb") as stream:
        for record_index, record in enumerate(ArchiveIterator(stream)):
            if record_index >= config.limit:
                break
            record_type = record.rec_type or "unknown"
            payload = record.content_stream().read()
            total_payload_bytes += len(payload)

            warc_mime = _normalized_mime(
                record.rec_headers.get_header("WARC-Identified-Payload-Type")
            )
            http_status = (
                record.http_headers.get_statuscode()
                if record.rec_type == "response" and record.http_headers
                else None
            )
            http_content_type_raw = (
                record.http_headers.get_header("Content-Type") if record.http_headers else None
            )
            http_mime, declared_charset = _http_content_type(http_content_type_raw)
            detected_mime, detected_encoding, encoding_confidence = _detect_payload(
                payload, magic_module
            )
            preview, preview_encoding, preview_decoded_strictly = _decode_preview(
                payload,
                declared_charset,
                detected_encoding,
                config.preview_chars,
            )
            datatrove_kept, datatrove_reason = _datatrove_standard_decision(
                record_type=record.rec_type,
                warc_mime=warc_mime,
                detected_mime=detected_mime,
                payload=payload,
            )

            record_types[record_type] += 1
            statuses[http_status or "none"] += 1
            warc_mimes[warc_mime or "none"] += 1
            http_mimes[http_mime or "none"] += 1
            detected_mimes[detected_mime or "none"] += 1
            datatrove_decisions[datatrove_reason] += 1
            records.append(
                {
                    "record_index": record_index,
                    "record_type": record_type,
                    "warc_record_id": record.rec_headers.get_header("WARC-Record-ID"),
                    "url": record.rec_headers.get_header("WARC-Target-URI"),
                    "warc_date": record.rec_headers.get_header("WARC-Date"),
                    "http_status": http_status,
                    "warc_identified_payload_type": warc_mime,
                    "http_content_type": http_content_type_raw,
                    "http_mime_type": http_mime,
                    "declared_charset": declared_charset,
                    "detected_mime_type": detected_mime,
                    "detected_encoding": detected_encoding,
                    "encoding_confidence": encoding_confidence,
                    "payload_bytes": len(payload),
                    "preview": preview,
                    "preview_encoding": preview_encoding,
                    "preview_decoded_strictly": preview_decoded_strictly,
                    "datatrove_standard_reader_would_keep": datatrove_kept,
                    "datatrove_standard_reader_decision": datatrove_reason,
                }
            )

    elapsed_seconds = time.perf_counter() - started
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
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
            "sha256": _sha256(input_path),
        },
        "probe": {
            "requested_record_limit": config.limit,
            "records_inspected": len(records),
            "preview_chars": config.preview_chars,
            "payload_bytes_inspected": total_payload_bytes,
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": len(records) / elapsed_seconds if elapsed_seconds else None,
        },
        "counts": {
            "record_types": _counter_dict(record_types),
            "http_statuses": _counter_dict(statuses),
            "warc_identified_payload_types": _counter_dict(warc_mimes),
            "http_mime_types": _counter_dict(http_mimes),
            "detected_mime_types": _counter_dict(detected_mimes),
            "datatrove_standard_reader_decisions": _counter_dict(datatrove_decisions),
        },
    }
    _write_jsonl_atomic(output_dir / "records.jsonl", records)
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary
