"""Distributed, document-separated Paloma statistics evaluation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EvaluationConfig
from .constants import SCHEMA_VERSION
from .documents import Segment, segment_document
from .metrics import Totals, merge_totals, summarize
from .provenance import installed_versions, verify_inventory


def _distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def _batched(items: Iterable[Segment], size: int) -> Iterator[list[Segment]]:
    batch: list[Segment] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _segments(
    *,
    dataset_path: Path,
    split: str,
    sources: list[str],
    tokenizer: Any,
    max_sequence_length: int,
    rank: int,
    world_size: int,
    max_documents: int | None,
) -> Iterator[Segment]:
    global_index = 0
    for source in sources:
        dataset = load_dataset(
            str(dataset_path),
            name=source,
            split=split,
            streaming=True,
        )
        for record in dataset:
            if max_documents is not None and global_index >= max_documents:
                return
            include = global_index % world_size == rank
            global_index += 1
            if not include:
                continue
            yield from segment_document(
                source=source,
                record=record,
                tokenizer=tokenizer,
                max_sequence_length=max_sequence_length,
            )


@torch.inference_mode()
def run_statistics(*, model_path: Path, config: EvaluationConfig, output_path: Path) -> Path:
    rank, world_size, device = _distributed()
    model_manifest_path = model_path / "evaluation_manifest.json"
    if not model_manifest_path.is_file():
        raise FileNotFoundError(f"Packaged model manifest missing: {model_manifest_path}")
    model_manifest = json.loads(model_manifest_path.read_text())
    verify_inventory(model_path, model_manifest["files"])
    asset_manifest_path = config.assets_dir / "paloma_manifest.json"
    if not asset_manifest_path.is_file():
        raise FileNotFoundError(f"Paloma asset manifest missing: {asset_manifest_path}")
    asset_manifest = json.loads(asset_manifest_path.read_text())
    if asset_manifest["revision"] != config.statistics.dataset_revision:
        raise ValueError("Paloma config revision does not match prepared asset manifest")
    verify_inventory(Path(asset_manifest["dataset_path"]), asset_manifest["files"])

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.bos_token_id is None:
        raise ValueError("Packaged tokenizer has no BOS token")
    dtype = getattr(torch, config.statistics.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    source_parts: dict[str, Totals] = {}
    domain_parts: dict[str, Totals] = {}
    dataset_path = Path(asset_manifest["dataset_path"])
    segments = _segments(
        dataset_path=dataset_path,
        split=config.statistics.split,
        sources=list(asset_manifest["sources"]),
        tokenizer=tokenizer,
        max_sequence_length=config.statistics.max_sequence_length,
        rank=rank,
        world_size=world_size,
        max_documents=config.statistics.max_documents,
    )
    started = time.monotonic()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    for batch in _batched(segments, config.statistics.batch_size):
        width = max(len(segment.token_ids) for segment in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row, segment in enumerate(batch):
            length = len(segment.token_ids)
            ids = torch.tensor(segment.token_ids, dtype=torch.long, device=device)
            input_ids[row, :length] = ids
            attention_mask[row, :length] = 1
            labels[row, 1:length] = ids[1:]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        token_losses = functional.cross_entropy(
            logits[:, :-1].float().transpose(1, 2),
            labels[:, 1:],
            reduction="none",
            ignore_index=-100,
        )
        for row, segment in enumerate(batch):
            valid = labels[row, 1:] != -100
            nll_nats = float(token_losses[row][valid].sum().item())
            predicted_tokens = int(valid.sum().item())
            for parts, key in (
                (source_parts, segment.source),
                (domain_parts, segment.domain),
            ):
                totals = parts.setdefault(key, Totals())
                totals.nll_nats += nll_nats
                totals.predicted_tokens += predicted_tokens
                totals.utf8_bytes += segment.document_bytes
                totals.documents += segment.document_count

    gathered: list[tuple[dict[str, Totals], dict[str, Totals]] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, (source_parts, domain_parts))
    else:
        gathered[0] = (source_parts, domain_parts)
    if rank == 0:
        present = [part for part in gathered if part is not None]
        merged_sources = merge_totals(part[0] for part in present)
        merged_domains = merge_totals(part[1] for part in present)
        source_summary = summarize(merged_sources)
        domain_summary = summarize(merged_domains)
        result = {
            "schema_version": SCHEMA_VERSION,
            "suite": "paloma_statistics",
            "model": model_manifest,
            "dataset": {
                "repo_id": asset_manifest["repo_id"],
                "revision": asset_manifest["revision"],
                "split": config.statistics.split,
            },
            "inference": {
                "max_sequence_length": config.statistics.max_sequence_length,
                "document_separated": True,
                "overlap": 0,
                "prepend_bos": True,
                "score_bos": False,
                "append_eos": False,
                "dtype": config.statistics.dtype,
                "world_size": world_size,
            },
            "elapsed_seconds": time.monotonic() - started,
            "environment": installed_versions(
                ("torch", "transformers", "datasets", "huggingface-hub")
            ),
            "micro": domain_summary["micro"],
            "macro": domain_summary["macro"],
            "sources": source_summary["domains"],
            "domains": domain_summary["domains"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if world_size > 1:
        dist.barrier()
    return output_path
