"""Exact corpus-level language-model statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass
class Totals:
    nll_nats: float = 0.0
    predicted_tokens: int = 0
    utf8_bytes: int = 0
    documents: int = 0

    def add(self, other: Totals) -> None:
        self.nll_nats += other.nll_nats
        self.predicted_tokens += other.predicted_tokens
        self.utf8_bytes += other.utf8_bytes
        self.documents += other.documents

    def derived(self) -> dict[str, float | int]:
        if self.predicted_tokens <= 0:
            raise ValueError("Cannot derive token metrics without predicted tokens")
        if self.utf8_bytes <= 0:
            raise ValueError("Cannot derive bits per byte without UTF-8 bytes")
        loss = self.nll_nats / self.predicted_tokens
        result: dict[str, float | int] = asdict(self)
        result.update(
            loss_nats_per_token=loss,
            perplexity=math.exp(loss),
            bits_per_token=loss / math.log(2),
            bits_per_byte=self.nll_nats / (math.log(2) * self.utf8_bytes),
        )
        return result


def merge_totals(parts: Iterable[dict[str, Totals]]) -> dict[str, Totals]:
    merged: dict[str, Totals] = {}
    for part in parts:
        for key, value in part.items():
            merged.setdefault(key, Totals()).add(value)
    return merged


def summarize(parts: dict[str, Totals]) -> dict[str, object]:
    """Create micro totals, per-domain metrics, and unweighted macro metrics."""

    if not parts:
        raise ValueError("Statistics evaluation produced no domains")
    micro = Totals()
    domains: dict[str, dict[str, float | int]] = {}
    for name in sorted(parts):
        micro.add(parts[name])
        domains[name] = parts[name].derived()
    metric_names = ("loss_nats_per_token", "perplexity", "bits_per_token", "bits_per_byte")
    macro = {
        metric: sum(float(value[metric]) for value in domains.values()) / len(domains)
        for metric in metric_names
    }
    return {"micro": micro.derived(), "macro": macro, "domains": domains}
