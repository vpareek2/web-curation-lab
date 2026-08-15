"""Stateful, rank-aware TorchTitan loading for DataTrove token shards."""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer

from web_curation_lab.tokenizer_assets import (
    EXPECTED_EOS_ID,
    EXPECTED_VOCAB_SIZE,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)

DATASET_FORMAT = "datatrove-ds"
VIEW_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_training_view(
    *,
    dataset_dir: Path,
    output_path: Path,
    sequence_length: int,
    global_batch_size: int,
    training_steps: int,
    seed: int,
    token_size_bytes: int = 2,
    eos_token_id: int = EXPECTED_EOS_ID,
) -> dict[str, Any]:
    """Freeze an ordered, batch-aligned view over existing DataTrove shards."""

    dataset_dir = dataset_dir.resolve()
    output_path = output_path.resolve()
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if training_steps <= 0:
        raise ValueError("training_steps must be positive")
    if token_size_bytes not in (2, 4):
        raise ValueError("token_size_bytes must be 2 or 4")

    shard_paths = sorted(dataset_dir.glob("*.ds"))
    if not shard_paths:
        raise FileNotFoundError(f"No .ds shards found in {dataset_dir}")
    ordering = np.random.default_rng(seed).permutation(len(shard_paths))
    shard_paths = [shard_paths[int(index)] for index in ordering]

    tokens_per_sample = sequence_length + 1
    shards: list[dict[str, Any]] = []
    available_samples = 0
    for shard_path in shard_paths:
        index_path = shard_path.with_suffix(shard_path.suffix + ".index")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing DataTrove index for {shard_path}: {index_path}")
        byte_count = shard_path.stat().st_size
        if byte_count % token_size_bytes:
            raise ValueError(
                f"Shard byte size is not divisible by token_size_bytes: {shard_path}"
            )
        token_count = byte_count // token_size_bytes
        sample_count = token_count // tokens_per_sample
        if sample_count == 0:
            raise ValueError(f"Shard contains no complete training samples: {shard_path}")
        available_samples += sample_count
        shards.append(
            {
                "path": os.path.relpath(shard_path, output_path.parent),
                "bytes": byte_count,
                "sha256": _sha256(shard_path),
                "index_bytes": index_path.stat().st_size,
                "index_sha256": _sha256(index_path),
                "tokens": token_count,
                "samples": sample_count,
            }
        )

    selected_samples = global_batch_size * training_steps
    if selected_samples > available_samples:
        raise ValueError(
            f"Training view requires {selected_samples} samples but only "
            f"{available_samples} complete samples are available"
        )
    value: dict[str, Any] = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "dataset_format": DATASET_FORMAT,
        "sequence_length": sequence_length,
        "token_size_bytes": token_size_bytes,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "eos_token_id": eos_token_id,
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "files_sha256": TOKENIZER_SHA256,
        },
        "seed": seed,
        "global_batch_size": global_batch_size,
        "training_steps": training_steps,
        "available_samples": available_samples,
        "available_predicted_tokens": available_samples * sequence_length,
        "selected_samples": selected_samples,
        "selected_predicted_tokens": selected_samples * sequence_length,
        "shards": shards,
    }
    _write_json_atomic(output_path, value)
    return value


@dataclass(frozen=True)
class _Shard:
    path: Path
    byte_count: int
    sha256: str
    samples: int


class DataTroveTrainingDataset(Dataset):
    """Map a frozen training view to shifted causal-LM examples."""

    def __init__(self, view_path: str | Path, *, verify_hashes: bool = False) -> None:
        from datatrove.utils.dataset import DatatroveFileDataset

        self.view_path = Path(view_path).resolve()
        if not self.view_path.is_file():
            raise FileNotFoundError(f"DataTrove training view does not exist: {self.view_path}")
        try:
            view = json.loads(self.view_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Could not read training view {self.view_path}: {error}") from error
        self.view = view
        if view.get("schema_version") != VIEW_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported training-view schema version: {view.get('schema_version')!r}"
            )
        if view.get("dataset_format") != DATASET_FORMAT:
            raise ValueError(f"Expected dataset_format={DATASET_FORMAT!r}")
        self.sequence_length = int(view["sequence_length"])
        self.token_size_bytes = int(view["token_size_bytes"])
        self.vocab_size = int(view["vocab_size"])
        self.eos_token_id = int(view["eos_token_id"])
        self.selected_samples = int(view["selected_samples"])
        self.global_batch_size = int(view["global_batch_size"])
        self.training_steps = int(view["training_steps"])
        if self.token_size_bytes not in (2, 4):
            raise ValueError("token_size_bytes must be 2 or 4")
        if self.vocab_size != EXPECTED_VOCAB_SIZE:
            raise ValueError(
                f"Training view vocab size {self.vocab_size} does not match "
                f"the pinned tokenizer vocab size {EXPECTED_VOCAB_SIZE}"
            )
        if self.eos_token_id != EXPECTED_EOS_ID:
            raise ValueError(
                f"Training view EOS ID {self.eos_token_id} does not match "
                f"the pinned tokenizer EOS ID {EXPECTED_EOS_ID}"
            )
        expected_tokenizer = {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "files_sha256": TOKENIZER_SHA256,
        }
        if view.get("tokenizer") != expected_tokenizer:
            raise ValueError(
                "Training-view tokenizer manifest does not match the pinned "
                f"{TOKENIZER_REPO_ID}@{TOKENIZER_REVISION} assets"
            )
        expected_selected = self.global_batch_size * self.training_steps
        if self.selected_samples != expected_selected:
            raise ValueError(
                f"selected_samples={self.selected_samples} does not equal "
                f"global_batch_size * training_steps={expected_selected}"
            )

        shards: list[_Shard] = []
        datasets = []
        cumulative_samples = [0]
        for raw_shard in view.get("shards", []):
            shard_path = (self.view_path.parent / raw_shard["path"]).resolve()
            if not shard_path.is_file():
                raise FileNotFoundError(f"DataTrove shard does not exist: {shard_path}")
            expected_bytes = int(raw_shard["bytes"])
            actual_bytes = shard_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"Shard size mismatch for {shard_path}: "
                    f"expected {expected_bytes}, found {actual_bytes}"
                )
            expected_hash = str(raw_shard["sha256"])
            if verify_hashes and _sha256(shard_path) != expected_hash:
                raise ValueError(f"Shard SHA-256 mismatch for {shard_path}")
            index_path = shard_path.with_suffix(shard_path.suffix + ".index")
            if not index_path.is_file():
                raise FileNotFoundError(f"DataTrove shard index does not exist: {index_path}")
            expected_index_bytes = int(raw_shard["index_bytes"])
            if index_path.stat().st_size != expected_index_bytes:
                raise ValueError(
                    f"Shard index size mismatch for {index_path}: expected "
                    f"{expected_index_bytes}, found {index_path.stat().st_size}"
                )
            if verify_hashes and _sha256(index_path) != str(raw_shard["index_sha256"]):
                raise ValueError(f"Shard index SHA-256 mismatch for {index_path}")
            dataset = DatatroveFileDataset(
                str(shard_path),
                seq_len=self.sequence_length,
                token_size=self.token_size_bytes,
                return_positions=True,
                positions_from_eos_token_id=self.eos_token_id,
                fsize=actual_bytes,
            )
            declared_samples = int(raw_shard.get("samples", len(dataset)))
            if declared_samples != len(dataset):
                raise ValueError(
                    f"Shard sample-count mismatch for {shard_path}: "
                    f"expected {declared_samples}, found {len(dataset)}"
                )
            shards.append(_Shard(shard_path, actual_bytes, expected_hash, len(dataset)))
            datasets.append(dataset)
            cumulative_samples.append(cumulative_samples[-1] + len(dataset))
        if not datasets:
            raise ValueError("Training view contains no shards")
        if self.selected_samples > cumulative_samples[-1]:
            raise ValueError(
                f"Training view selects {self.selected_samples} samples but its shards "
                f"contain only {cumulative_samples[-1]}"
            )
        self.shards = shards
        self._datasets = datasets
        self._cumulative_samples = cumulative_samples
        self._bulk_handles: dict[int, BinaryIO] = {}

    def __len__(self) -> int:
        return self.selected_samples

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        shard_index = bisect_right(self._cumulative_samples, item) - 1
        local_item = item - self._cumulative_samples[shard_index]
        sample = self._datasets[shard_index][local_item]
        tokens = sample["input_ids"]
        positions = sample["positions"]
        if tokens.numel() != self.sequence_length + 1:
            raise ValueError(
                f"Expected {self.sequence_length + 1} tokens, found {tokens.numel()}"
            )
        return {
            "input": tokens[:-1],
            "positions": positions[:-1],
        }, tokens[1:]

    def _positions_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Vectorize DataTrove's per-sample EOS position-reset contract."""

        rows, width = tokens.shape
        flat = tokens.reshape(-1)
        offsets = torch.arange(flat.numel(), dtype=torch.long)
        starts = torch.zeros(flat.numel(), dtype=torch.bool)
        starts[::width] = True
        starts[1:] |= flat[:-1] == self.eos_token_id
        anchors = torch.where(starts, offsets, 0)
        last_anchor = torch.cummax(anchors, dim=0).values
        return (offsets - last_anchor).reshape(rows, width)

    def _read_contiguous(
        self,
        *,
        shard_index: int,
        first_local_item: int,
        count: int,
    ) -> list[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        handle = self._bulk_handles.get(shard_index)
        if handle is None:
            handle = self.shards[shard_index].path.open("rb")
            self._bulk_handles[shard_index] = handle
        tokens_per_sample = self.sequence_length + 1
        bytes_per_sample = tokens_per_sample * self.token_size_bytes
        handle.seek(first_local_item * bytes_per_sample)
        raw = handle.read(count * bytes_per_sample)
        if len(raw) != count * bytes_per_sample:
            raise ValueError(
                f"Short read from {self.shards[shard_index].path}: expected "
                f"{count * bytes_per_sample} bytes, found {len(raw)}"
            )
        dtype = np.dtype("<u2" if self.token_size_bytes == 2 else "<u4")
        token_array = np.frombuffer(raw, dtype=dtype).reshape(count, tokens_per_sample)
        tokens = torch.as_tensor(token_array.astype(np.int64), dtype=torch.long)
        positions = self._positions_from_tokens(tokens)
        return [
            (
                {
                    "input": tokens[row, :-1],
                    "positions": positions[row, :-1],
                },
                tokens[row, 1:],
            )
            for row in range(count)
        ]

    def __getitems__(
        self, items: list[int]
    ) -> list[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Fetch each contiguous batch with one read per intersected shard."""

        normalized = [int(item) for item in items]
        for item in normalized:
            if not 0 <= item < len(self):
                raise IndexError(item)
        output: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
        cursor = 0
        while cursor < len(normalized):
            item = normalized[cursor]
            shard_index = bisect_right(self._cumulative_samples, item) - 1
            first_local_item = item - self._cumulative_samples[shard_index]
            run_length = 1
            while cursor + run_length < len(normalized):
                next_item = normalized[cursor + run_length]
                next_shard = bisect_right(self._cumulative_samples, next_item) - 1
                next_local_item = next_item - self._cumulative_samples[next_shard]
                if (
                    next_shard != shard_index
                    or next_local_item != first_local_item + run_length
                ):
                    break
                run_length += 1
            output.extend(
                self._read_contiguous(
                    shard_index=shard_index,
                    first_local_item=first_local_item,
                    count=run_length,
                )
            )
            cursor += run_length
        return output

    def __del__(self) -> None:
        for handle in getattr(self, "_bulk_handles", {}).values():
            handle.close()


class _RankBatchSampler(Sampler[list[int]]):
    """Assign one contiguous slice of each global batch to a DDP rank."""

    def __init__(
        self,
        *,
        training_steps: int,
        local_batch_size: int,
        dp_rank: int,
        dp_world_size: int,
    ) -> None:
        self.training_steps = training_steps
        self.local_batch_size = local_batch_size
        self.dp_rank = dp_rank
        self.global_batch_size = local_batch_size * dp_world_size

    def __iter__(self):
        rank_offset = self.dp_rank * self.local_batch_size
        for step in range(self.training_steps):
            start = step * self.global_batch_size + rank_offset
            yield list(range(start, start + self.local_batch_size))

    def __len__(self) -> int:
        return self.training_steps


class DataTroveTokenDataLoader(ParallelAwareDataloader):
    """TorchTitan dataloader for a frozen DataTrove training view."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "datatrove"
        verify_hashes: bool = True

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ) -> None:
        del tokenizer, kwargs
        if not config.dataset_path:
            raise ValueError("DataTroveTokenDataLoader requires dataloader.dataset_path")
        dataset = DataTroveTrainingDataset(
            config.dataset_path,
            # Every rank validates sizes and structure. One data-parallel rank
            # performs the expensive content-hash pass so startup reads each
            # shard once rather than once per GPU.
            verify_hashes=config.verify_hashes and dp_rank == 0,
        )
        if dataset.sequence_length != seq_len:
            raise ValueError(
                f"Training view sequence length {dataset.sequence_length} does not match "
                f"TorchTitan sequence length {seq_len}"
            )
        actual_global_batch_size = local_batch_size * dp_world_size
        if dataset.global_batch_size != actual_global_batch_size:
            raise ValueError(
                f"Training view global batch size {dataset.global_batch_size} does not match "
                f"TorchTitan global batch size {actual_global_batch_size}"
            )
        batch_sampler = _RankBatchSampler(
            training_steps=dataset.training_steps,
            local_batch_size=local_batch_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "snapshot_every_n_steps": snapshot_every_n_steps,
            "batch_sampler": batch_sampler,
        }
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
