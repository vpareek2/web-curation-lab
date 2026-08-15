"""DataTrove reader for minimally processed textual WARC response payloads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datatrove.pipeline.readers.base import BaseDiskReader

from web_curation_lab.pipeline.stages.raw_cc.probe import (
    _detect_payload,
    _http_content_type,
    _load_magic,
    _normalized_mime,
)

if TYPE_CHECKING:
    from datatrove.data import Document


@dataclass(frozen=True)
class PayloadDecision:
    """Mechanical MIME and decoding decision shared by census and audit paths."""

    warc_mime: str | None
    http_content_type: str | None
    http_mime: str | None
    declared_charset: str | None
    detected_mime: str | None
    detected_encoding: str | None
    encoding_confidence: float | None
    effective_mime: str | None
    is_textual_mime: bool
    text: str | None
    decode_encoding: str | None
    decode_used_replacement: bool
    replacement_characters: int


class RawWarcReader(BaseDiskReader):
    """Yield decoded textual HTTP responses while accounting for every WARC record.

    MIME selection is deliberately configurable and provisional. No HTML
    extraction, normalization, language filtering, deduplication, or quality
    filtering occurs here.
    """

    name = "Raw textual WARC responses"
    _requires_dependencies = ["warcio", ("cchardet", "faust-cchardet")]

    def __init__(
        self,
        data_folder: str,
        *,
        crawl_id: str,
        policy_revision: str,
        text_mime_prefixes: Iterable[str],
        text_mime_types: Iterable[str],
        text_mime_suffixes: Iterable[str],
        glob_pattern: str | None = None,
    ) -> None:
        super().__init__(
            data_folder,
            glob_pattern=glob_pattern,
            recursive=False,
            default_metadata={
                "crawl_id": crawl_id,
                "stage_id": "0_raw_cc",
                "policy_revision": policy_revision,
            },
        )
        self.crawl_id = crawl_id
        self.policy_revision = policy_revision
        self.text_mime_prefixes = tuple(value.lower() for value in text_mime_prefixes)
        self.text_mime_types = frozenset(value.lower() for value in text_mime_types)
        self.text_mime_suffixes = tuple(value.lower() for value in text_mime_suffixes)
        magic_module, self.magic_error = _load_magic()
        self.libmagic_available = magic_module is not None

    def _is_textual_mime(self, mime_type: str | None) -> bool:
        if mime_type is None:
            return False
        return (
            mime_type in self.text_mime_types
            or mime_type.startswith(self.text_mime_prefixes)
            or mime_type.endswith(self.text_mime_suffixes)
        )

    def _decode_payload(
        self,
        payload: bytes,
        declared_charset: str | None,
        detected_encoding: str | None,
    ) -> tuple[str, str, bool, int]:
        candidates = [declared_charset, "utf-8", detected_encoding]
        tried: set[str] = set()
        valid_candidates: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            normalized = candidate.lower()
            if normalized in tried:
                continue
            tried.add(normalized)
            try:
                text = payload.decode(candidate)
                return text, candidate, False, 0
            except LookupError:
                continue
            except UnicodeDecodeError:
                valid_candidates.append(candidate)
        for candidate in valid_candidates:
            text = payload.decode(candidate, errors="replace")
            return text, candidate, True, text.count("\ufffd")
        # UTF-8 is always among the candidates, so this is defensive only.
        raise RuntimeError("No valid text decoder was available, including UTF-8")

    def _count_value(self, namespace: str, value: str | None) -> None:
        self.stat_update(f"{namespace}::{value or 'none'}")

    def inspect_payload(
        self,
        *,
        payload: bytes,
        warc_identified_payload_type: str | None,
        http_content_type: str | None,
        magic_module: object | None,
    ) -> PayloadDecision:
        """Apply the reader's exact provisional MIME and best-effort decode policy."""

        warc_mime = _normalized_mime(warc_identified_payload_type)
        http_mime, declared_charset = _http_content_type(http_content_type)
        detected_mime, detected_encoding, encoding_confidence = _detect_payload(
            payload, magic_module
        )
        effective_mime = warc_mime or http_mime or detected_mime
        is_textual_mime = self._is_textual_mime(effective_mime)
        text: str | None = None
        decode_encoding: str | None = None
        decode_used_replacement = False
        replacement_characters = 0
        if is_textual_mime:
            (
                text,
                decode_encoding,
                decode_used_replacement,
                replacement_characters,
            ) = self._decode_payload(
                payload,
                declared_charset,
                detected_encoding,
            )
        return PayloadDecision(
            warc_mime=warc_mime,
            http_content_type=http_content_type,
            http_mime=http_mime,
            declared_charset=declared_charset,
            detected_mime=detected_mime,
            detected_encoding=detected_encoding,
            encoding_confidence=encoding_confidence,
            effective_mime=effective_mime,
            is_textual_mime=is_textual_mime,
            text=text,
            decode_encoding=decode_encoding,
            decode_used_replacement=decode_used_replacement,
            replacement_characters=replacement_characters,
        )

    def read_file(self, filepath: str) -> Iterable[Document]:
        from warcio.archiveiterator import ArchiveIterator

        magic_module, _ = _load_magic()
        if magic_module is None:
            self.stat_update("libmagic_unavailable")
        with self.data_folder.open(filepath, "rb", compression="infer") as stream:
            for record_index, record in enumerate(ArchiveIterator(stream)):
                record_type = record.rec_type or "unknown"
                self.stat_update("warc_records")
                self._count_value("record_type", record_type)
                if record.rec_type != "response":
                    continue

                self.stat_update("response_records")
                http_status = record.http_headers.get_statuscode() if record.http_headers else None
                self._count_value("http_status", http_status)
                payload = record.content_stream().read()
                self.stat_update("response_payload_bytes", value=len(payload), unit="response")

                http_content_type = (
                    record.http_headers.get_header("Content-Type") if record.http_headers else None
                )
                decision = self.inspect_payload(
                    payload=payload,
                    warc_identified_payload_type=record.rec_headers.get_header(
                        "WARC-Identified-Payload-Type"
                    ),
                    http_content_type=http_content_type,
                    magic_module=magic_module,
                )
                self._count_value("warc_mime", decision.warc_mime)
                self._count_value("http_mime", decision.http_mime)
                self._count_value("detected_mime", decision.detected_mime)
                self._count_value("effective_mime", decision.effective_mime)
                if (
                    decision.warc_mime
                    and decision.http_mime
                    and decision.warc_mime != decision.http_mime
                ):
                    self.stat_update("warc_http_mime_disagreements")
                if not decision.is_textual_mime:
                    self.stat_update("dropped_non_text_mime")
                    continue

                self.stat_update("textual_mime_candidates")
                if decision.text is None:
                    self.stat_update("dropped_decode_failed")
                    continue
                if decision.decode_used_replacement:
                    self.stat_update("decoded_with_replacement")
                    self.stat_update(
                        "replacement_characters",
                        value=decision.replacement_characters,
                        unit="document",
                    )
                if not decision.text:
                    self.stat_update("dropped_empty_text")
                    continue

                record_id = record.rec_headers.get_header("WARC-Record-ID")
                if not record_id:
                    self.stat_update("missing_warc_record_id")
                    record_id = f"{self.crawl_id}:{filepath}:{record_index}"
                self.stat_update("decoded_documents")
                self._count_value("decode_encoding", decision.decode_encoding)
                document = self.get_document_from_dict(
                    {
                        "id": record_id,
                        "text": decision.text,
                        "url": record.rec_headers.get_header("WARC-Target-URI"),
                        "warc_date": record.rec_headers.get_header("WARC-Date"),
                        "http_status": http_status,
                        "warc_identified_payload_type": decision.warc_mime,
                        "http_content_type": http_content_type,
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
                        "source_record_index": record_index,
                    },
                    filepath,
                    record_index,
                )
                if document is not None:
                    yield document
