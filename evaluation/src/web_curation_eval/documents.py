"""Paloma document normalization and non-overlapping segmentation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

DOMAIN_KEYS = (
    "domain",
    "subdomain",
    "subreddit",
    "programming_language",
    "language",
    "category",
)


@dataclass(frozen=True)
class Segment:
    source: str
    domain: str
    token_ids: list[int]
    document_bytes: int
    document_count: int


def infer_domain(source: str, record: dict[str, Any]) -> str:
    for key in DOMAIN_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return f"{source}/{value.strip()}"
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in DOMAIN_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return f"{source}/{value.strip()}"
    return source


def segment_document(
    *,
    source: str,
    record: dict[str, Any],
    tokenizer: Any,
    max_sequence_length: int,
) -> Iterator[Segment]:
    """Yield BOS-prefixed, disjoint segments while counting the document once."""

    text = record.get("text")
    if not isinstance(text, str) or not text:
        return
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("Statistics evaluation requires a BOS token")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return
    capacity = max_sequence_length - 1
    domain = infer_domain(source, record)
    for index, start in enumerate(range(0, len(tokens), capacity)):
        yield Segment(
            source=source,
            domain=domain,
            token_ids=[bos_id, *tokens[start : start + capacity]],
            document_bytes=len(text.encode("utf-8")) if index == 0 else 0,
            document_count=1 if index == 0 else 0,
        )
