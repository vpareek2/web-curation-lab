"""Full-file counts for the provisional ``0_raw_cc`` inclusion contract."""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens import TokensCounter
from datatrove.utils.stats import MetricStats, PipelineStats

from web_curation_lab.pipeline.stages.raw_cc.probe import STAGE_ID
from web_curation_lab.pipeline.stages.raw_cc.reader import RawWarcReader
from web_curation_lab.tokenizer_assets import verify_tokenizer


@dataclass(frozen=True)
class CensusConfig:
    stage_id: str
    crawl_id: str
    policy_revision: str
    input_path: Path
    output_dir: Path
    tokenizer_path: Path
    tokenizer_batch_size: int
    text_mime_prefixes: tuple[str, ...]
    text_mime_types: tuple[str, ...]
    text_mime_suffixes: tuple[str, ...]


class CensusCounters(PipelineStep):
    """Terminal DataTrove block for decoded-byte and token totals."""

    name = "Raw CC census totals"
    type = "Stats"

    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1,
    ) -> DocumentsPipeline:
        del rank, world_size
        if data is None:
            return
        for document in data:
            self.stat_update("documents")
            self.stat_update(
                "source_payload_bytes",
                value=int(document.metadata["payload_bytes"]),
                unit="document",
            )
            self.stat_update(
                "decoded_utf8_bytes",
                value=len(document.text.encode("utf-8")),
                unit="document",
            )
            self.stat_update(
                "content_tokens",
                value=int(document.metadata["token_count"]),
                unit="document",
            )
            yield document


def load_census_config(config_path: Path) -> CensusConfig:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        stage = raw["stage"]
        probe = raw["probe"]
        census = raw["census"]
        config = CensusConfig(
            stage_id=str(stage["id"]),
            crawl_id=str(stage["crawl_id"]),
            policy_revision=str(stage["policy_revision"]),
            input_path=Path(probe["input_path"]),
            output_dir=Path(census["output_dir"]),
            tokenizer_path=Path(census["tokenizer_path"]),
            tokenizer_batch_size=int(census["tokenizer_batch_size"]),
            text_mime_prefixes=tuple(census["text_mime_prefixes"]),
            text_mime_types=tuple(census["text_mime_types"]),
            text_mime_suffixes=tuple(census["text_mime_suffixes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid raw-CC census config {config_path}: {error}") from error
    if config.stage_id != STAGE_ID:
        raise ValueError(f"Expected stage.id={STAGE_ID!r}, got {config.stage_id!r}")
    if config.tokenizer_batch_size <= 0:
        raise ValueError("census.tokenizer_batch_size must be positive")
    if not (config.text_mime_prefixes or config.text_mime_types or config.text_mime_suffixes):
        raise ValueError("The provisional textual MIME policy cannot be empty")
    if not config.policy_revision:
        raise ValueError("stage.policy_revision cannot be empty")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_total(metrics: dict[str, MetricStats], name: str) -> int:
    metric = metrics.get(name)
    return int(metric.total) if metric is not None else 0


def _metric_namespace(metrics: dict[str, MetricStats], namespace: str) -> dict[str, int]:
    prefix = f"{namespace}::"
    return dict(
        sorted(
            (name.removeprefix(prefix), int(metric.total))
            for name, metric in metrics.items()
            if name.startswith(prefix)
        )
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _build_summary(
    config: CensusConfig,
    input_path: Path,
    pipeline_stats: PipelineStats,
    wall_seconds: float,
) -> dict[str, Any]:
    reader_metrics = pipeline_stats.stats[0].stats
    counter_metrics = pipeline_stats.stats[1].stats
    census_metrics = pipeline_stats.stats[2].stats
    responses = _metric_total(reader_metrics, "response_records")
    documents = _metric_total(census_metrics, "documents")
    content_tokens = _metric_total(counter_metrics, "tokens")
    training_tokens = content_tokens + documents
    return {
        "schema_version": 1,
        "stage_id": config.stage_id,
        "crawl_id": config.crawl_id,
        "policy_revision": config.policy_revision,
        "policy_status": "mechanical reader policy frozen for multi-file pilot",
        "dependencies": {
            "datatrove": version("datatrove"),
            "tokenizers": version("tokenizers"),
            "warcio": version("warcio"),
        },
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
        },
        "textual_mime_policy": {
            "prefixes": list(config.text_mime_prefixes),
            "types": list(config.text_mime_types),
            "suffixes": list(config.text_mime_suffixes),
        },
        "counts": {
            "warc_records": _metric_total(reader_metrics, "warc_records"),
            "record_types": _metric_namespace(reader_metrics, "record_type"),
            "response_records": responses,
            "http_statuses": _metric_namespace(reader_metrics, "http_status"),
            "warc_mime_types": _metric_namespace(reader_metrics, "warc_mime"),
            "http_mime_types": _metric_namespace(reader_metrics, "http_mime"),
            "detected_mime_types": _metric_namespace(reader_metrics, "detected_mime"),
            "effective_mime_types": _metric_namespace(reader_metrics, "effective_mime"),
            "warc_http_mime_disagreements": _metric_total(
                reader_metrics, "warc_http_mime_disagreements"
            ),
            "response_payload_bytes": _metric_total(reader_metrics, "response_payload_bytes"),
            "textual_mime_candidates": _metric_total(reader_metrics, "textual_mime_candidates"),
            "dropped_non_text_mime": _metric_total(reader_metrics, "dropped_non_text_mime"),
            "dropped_decode_failed": _metric_total(reader_metrics, "dropped_decode_failed"),
            "decoded_with_replacement": _metric_total(
                reader_metrics, "decoded_with_replacement"
            ),
            "replacement_characters": _metric_total(
                reader_metrics, "replacement_characters"
            ),
            "dropped_empty_text": _metric_total(reader_metrics, "dropped_empty_text"),
            "decoded_documents": documents,
            "decode_encodings": _metric_namespace(reader_metrics, "decode_encoding"),
            "decoded_utf8_bytes": _metric_total(census_metrics, "decoded_utf8_bytes"),
            "content_tokens": content_tokens,
            "document_boundary_tokens": documents,
            "training_tokens_with_eos": training_tokens,
        },
        "ratios": {
            "decoded_documents_per_response": documents / responses if responses else None,
            "content_tokens_per_response_payload_byte": (
                content_tokens / _metric_total(reader_metrics, "response_payload_bytes")
                if _metric_total(reader_metrics, "response_payload_bytes")
                else None
            ),
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "compressed_input_bytes_per_second": (
                input_path.stat().st_size / wall_seconds if wall_seconds else None
            ),
        },
        "datatrove_stats": json.loads(pipeline_stats.to_json()),
    }


def run_census(config: CensusConfig) -> dict[str, Any]:
    input_path = config.input_path.resolve()
    output_dir = config.output_dir.resolve()
    tokenizer_path = config.tokenizer_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"WARC input does not exist: {input_path}")
    verify_tokenizer(tokenizer_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = [
        RawWarcReader(
            str(input_path.parent),
            crawl_id=config.crawl_id,
            policy_revision=config.policy_revision,
            glob_pattern=input_path.name,
            text_mime_prefixes=config.text_mime_prefixes,
            text_mime_types=config.text_mime_types,
            text_mime_suffixes=config.text_mime_suffixes,
        ),
        TokensCounter(
            tokenizer_name_or_path=str(tokenizer_path),
            count_eos_token=False,
            batch_size=config.tokenizer_batch_size,
        ),
        CensusCounters(),
    ]
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        workers=1,
        logging_dir=str(output_dir / "datatrove"),
        skip_completed=False,
    )
    started = time.perf_counter()
    pipeline_stats = executor.run()
    wall_seconds = time.perf_counter() - started
    if pipeline_stats is None:
        raise RuntimeError("DataTrove census returned no pipeline statistics")
    summary = _build_summary(config, input_path, pipeline_stats, wall_seconds)
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary
