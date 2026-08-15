from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

pytest.importorskip("datatrove")

from web_curation_lab.pipeline.cli import main as data_cli_main
from web_curation_lab.tokenizer_assets import (
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)
from web_curation_lab.training.datatrove_loader import (
    DataTroveTokenDataLoader,
    DataTroveTrainingDataset,
    create_training_view,
)


def _write_shard(path: Path, tokens: list[int], doc_ends: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    path.with_suffix(path.suffix + ".index").write_bytes(
        struct.pack(f"<{len(doc_ends)}Q", *doc_ends)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_view(
    path: Path,
    shards: list[Path],
    *,
    sequence_length: int,
    global_batch_size: int,
    training_steps: int,
) -> None:
    value = {
        "schema_version": 1,
        "dataset_format": "datatrove-ds",
        "sequence_length": sequence_length,
        "token_size_bytes": 2,
        "vocab_size": 32_000,
        "eos_token_id": 2,
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "files_sha256": TOKENIZER_SHA256,
        },
        "global_batch_size": global_batch_size,
        "training_steps": training_steps,
        "selected_samples": global_batch_size * training_steps,
        "selected_predicted_tokens": (
            global_batch_size * training_steps * sequence_length
        ),
        "shards": [
            {
                "path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": _sha256(shard),
                "index_bytes": shard.with_suffix(
                    shard.suffix + ".index"
                ).stat().st_size,
                "index_sha256": _sha256(shard.with_suffix(shard.suffix + ".index")),
            }
            for shard in shards
        ],
    }
    path.write_text(json.dumps(value))


def test_dataset_shifts_tokens_and_resets_positions_after_eos(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, [10, 11, 2, 20, 21, 22, 2, 30, 31, 32], [3, 7, 10])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=2,
        training_steps=1,
    )

    dataset = DataTroveTrainingDataset(view, verify_hashes=True)

    inputs, labels = dataset[0]
    assert inputs["input"].tolist() == [10, 11, 2, 20]
    assert labels.tolist() == [11, 2, 20, 21]
    assert inputs["positions"].tolist() == [0, 1, 2, 0]


def test_batched_fetch_matches_scalar_fetch_across_shards(tmp_path: Path) -> None:
    shard_a = tmp_path / "000.ds"
    shard_b = tmp_path / "001.ds"
    _write_shard(shard_a, [10, 11, 2, 20, 21, 22, 23, 24, 2, 25], [3, 9, 10])
    _write_shard(shard_b, [30, 2, 31, 32, 33, 40, 41, 42, 43, 44], [2, 10])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard_a, shard_b],
        sequence_length=4,
        global_batch_size=4,
        training_steps=1,
    )
    dataset = DataTroveTrainingDataset(view)
    expected = [dataset[index] for index in (1, 2)]

    for shard_dataset in dataset._datasets:
        shard_dataset.__getitem__ = lambda _: (_ for _ in ()).throw(
            AssertionError("batched fetch fell back to scalar DataTrove reads")
        )
    actual = dataset.__getitems__([1, 2])

    for (expected_inputs, expected_labels), (actual_inputs, actual_labels) in zip(
        expected, actual, strict=True
    ):
        assert torch.equal(expected_inputs["input"], actual_inputs["input"])
        assert torch.equal(expected_inputs["positions"], actual_inputs["positions"])
        assert torch.equal(expected_labels, actual_labels)


def test_two_ranks_cover_each_global_batch_once(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(100)), [100])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=4,
        training_steps=1,
    )

    loaders = [
        DataTroveTokenDataLoader(
            DataTroveTokenDataLoader.Config(dataset_path=str(view)),
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=object(),
            seq_len=4,
            local_batch_size=2,
        )
        for rank in range(2)
    ]
    rank_batches = [next(iter(loader))[0]["input"][:, 0].tolist() for loader in loaders]

    assert rank_batches == [[0, 5], [10, 15]]
    assert sorted(rank_batches[0] + rank_batches[1]) == [0, 5, 10, 15]


def test_dataloader_resume_continues_at_exact_next_batch(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(49)), [49])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=2,
        global_batch_size=2,
        training_steps=8,
    )
    config = DataTroveTokenDataLoader.Config(dataset_path=str(view))
    loader = DataTroveTokenDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=2,
        local_batch_size=2,
        snapshot_every_n_steps=1,
    )
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    uninterrupted = next(iterator)

    resumed = DataTroveTokenDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=2,
        local_batch_size=2,
        snapshot_every_n_steps=1,
    )
    resumed.load_state_dict(state)
    resumed_batch = next(iter(resumed))

    assert torch.equal(uninterrupted[0]["input"], resumed_batch[0]["input"])
    assert torch.equal(uninterrupted[1], resumed_batch[1])


def test_multiworker_loader_preserves_declared_batch_order(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(100)), [100])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=4,
        training_steps=4,
    )
    loader = DataTroveTokenDataLoader(
        DataTroveTokenDataLoader.Config(
            dataset_path=str(view),
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
        ),
        dp_world_size=2,
        dp_rank=1,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=2,
        snapshot_every_n_steps=1,
    )

    starts = [batch[0]["input"][:, 0].tolist() for batch in loader]

    assert starts == [[10, 15], [30, 35], [50, 55], [70, 75]]


def test_multiworker_dataloader_resume_is_exact(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(100)), [100])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=2,
        training_steps=10,
    )
    config = DataTroveTokenDataLoader.Config(
        dataset_path=str(view),
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    loader = DataTroveTokenDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=2,
        snapshot_every_n_steps=1,
    )
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    uninterrupted = next(iterator)

    resumed = DataTroveTokenDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=2,
        snapshot_every_n_steps=1,
    )
    resumed.load_state_dict(state)
    resumed_batch = next(iter(resumed))

    assert torch.equal(uninterrupted[0]["input"], resumed_batch[0]["input"])
    assert torch.equal(uninterrupted[1], resumed_batch[1])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sequence_length", 8, "sequence length"),
        ("global_batch_size", 8, "global batch size"),
        ("selected_samples", 3, "selected_samples"),
    ],
)
def test_loader_rejects_mismatched_view(
    tmp_path: Path, field: str, value: int, match: str
) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(100)), [100])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=4,
        training_steps=1,
    )
    raw = json.loads(view.read_text())
    raw[field] = value
    if field == "global_batch_size":
        raw["selected_samples"] = value * raw["training_steps"]
    view.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=match):
        DataTroveTokenDataLoader(
            DataTroveTokenDataLoader.Config(dataset_path=str(view)),
            dp_world_size=2,
            dp_rank=0,
            tokenizer=object(),
            seq_len=4,
            local_batch_size=2,
        )


def test_loader_rejects_truncated_or_corrupted_shard(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(10)), [10])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=2,
        training_steps=1,
    )
    shard.write_bytes(shard.read_bytes()[:-2])

    with pytest.raises(ValueError, match="size mismatch"):
        DataTroveTrainingDataset(view)


def test_loader_optional_hash_verification_detects_same_size_corruption(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(10)), [10])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=2,
        training_steps=1,
    )
    contents = bytearray(shard.read_bytes())
    contents[0] ^= 1
    shard.write_bytes(contents)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        DataTroveTrainingDataset(view, verify_hashes=True)


def test_loader_rejects_tokenizer_manifest_mismatch(tmp_path: Path) -> None:
    shard = tmp_path / "000.ds"
    _write_shard(shard, list(range(10)), [10])
    view = tmp_path / "view.json"
    _write_view(
        view,
        [shard],
        sequence_length=4,
        global_batch_size=2,
        training_steps=1,
    )
    raw = json.loads(view.read_text())
    raw["tokenizer"]["revision"] = "wrong"
    view.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="tokenizer manifest"):
        DataTroveTrainingDataset(view)


def test_create_training_view_is_deterministic_and_batch_aligned(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_a = shard_dir / "a.ds"
    shard_b = shard_dir / "b.ds"
    _write_shard(shard_a, list(range(15)), [15])
    _write_shard(shard_b, list(range(15, 30)), [15])

    first = create_training_view(
        dataset_dir=shard_dir,
        output_path=tmp_path / "views" / "view-a.json",
        sequence_length=4,
        global_batch_size=2,
        training_steps=3,
        seed=42,
    )
    second = create_training_view(
        dataset_dir=shard_dir,
        output_path=tmp_path / "views" / "view-b.json",
        sequence_length=4,
        global_batch_size=2,
        training_steps=3,
        seed=42,
    )

    assert first["shards"] == second["shards"]
    assert first["available_samples"] == 6
    assert first["selected_samples"] == 6
    assert first["selected_predicted_tokens"] == 24


def test_create_training_view_fails_when_data_is_short(tmp_path: Path) -> None:
    shard = tmp_path / "only.ds"
    _write_shard(shard, list(range(10)), [10])

    with pytest.raises(ValueError, match="requires 4 samples.*only 2"):
        create_training_view(
            dataset_dir=tmp_path,
            output_path=tmp_path / "view.json",
            sequence_length=4,
            global_batch_size=2,
            training_steps=2,
            seed=42,
        )


def test_create_training_view_cli_writes_loadable_view(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "only.ds"
    _write_shard(shard, list(range(20)), [20])
    view = tmp_path / "views" / "pilot.json"

    data_cli_main(
        [
            "create-training-view",
            "--dataset-dir",
            str(shard_dir),
            "--output",
            str(view),
            "--sequence-length",
            "4",
            "--global-batch-size",
            "2",
            "--training-steps",
            "2",
            "--seed",
            "7",
        ]
    )

    dataset = DataTroveTrainingDataset(view, verify_hashes=True)
    assert len(dataset) == 4
