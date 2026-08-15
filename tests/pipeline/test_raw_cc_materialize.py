from __future__ import annotations

import hashlib
import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datatrove")
pytest.importorskip("warcio")
from tokenizers import Tokenizer
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from web_curation_lab.pipeline.cli import main
from web_curation_lab.pipeline.stages.raw_cc.census import (
    load_census_config,
    run_census,
)
from web_curation_lab.pipeline.stages.raw_cc.materialize import (
    assemble_token_streams,
    load_materialize_config,
    run_materialization,
)
from web_curation_lab.training.datatrove_loader import (
    DataTroveTrainingDataset,
    create_training_view,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tokens(path: Path, tokens: list[int], document_ends: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    path.with_suffix(path.suffix + ".index").write_bytes(
        np.asarray(document_ends, dtype="<u8").tobytes()
    )


def _read_tokens(path: Path) -> list[int]:
    return np.frombuffer(path.read_bytes(), dtype="<u2").astype(int).tolist()


def test_assembler_packs_across_sources_and_discards_only_final_partial_sample(
    tmp_path: Path,
) -> None:
    first = tmp_path / "inputs" / "first.ds"
    second = tmp_path / "inputs" / "second.ds"
    _write_tokens(first, [10, 2, 11, 12, 2], [2, 5])
    _write_tokens(second, [20, 21, 2, 22, 23, 24, 2, 2], [3, 7, 8])

    summary = assemble_token_streams(
        input_paths=[first, second],
        input_index_paths=[first.with_suffix(".ds.index"), second.with_suffix(".ds.index")],
        output_dir=tmp_path / "shards",
        sequence_length=3,
        samples_per_shard=2,
        eos_token_id=2,
        token_size_bytes=2,
    )

    shards = sorted((tmp_path / "shards").glob("*.ds"))
    assert [_read_tokens(path) for path in shards] == [
        [10, 2, 11, 12, 2, 20, 21, 2],
        [22, 23, 24, 2],
    ]
    assert np.frombuffer(shards[0].with_suffix(".ds.index").read_bytes(), dtype="<u8").tolist() == [
        2,
        5,
        8,
    ]
    assert np.frombuffer(
        shards[1].with_suffix(".ds.index").read_bytes(), dtype="<u8"
    ).tolist() == [4]
    assert summary["counts"] == {
        "source_tokens": 13,
        "materialized_tokens": 12,
        "discarded_final_tokens": 1,
        "samples": 3,
        "document_boundaries": 4,
        "shards": 2,
    }
    assert all(len(shard["sha256"]) == 64 for shard in summary["shards"])
    assert all(shard["tokens"] % 4 == 0 for shard in summary["shards"])


def test_assembler_exact_boundary_does_not_leave_an_empty_shard(tmp_path: Path) -> None:
    source = tmp_path / "source.ds"
    _write_tokens(source, [1, 2, 3, 2, 4, 5, 6, 2], [4, 8])

    summary = assemble_token_streams(
        input_paths=[source],
        input_index_paths=[source.with_suffix(".ds.index")],
        output_dir=tmp_path / "shards",
        sequence_length=3,
        samples_per_shard=2,
        eos_token_id=2,
        token_size_bytes=2,
    )

    assert summary["counts"]["shards"] == 1
    assert summary["counts"]["discarded_final_tokens"] == 0
    assert [path.name for path in (tmp_path / "shards").glob("*.ds")] == ["00000.ds"]


def _write_response(writer: WARCWriter, *, url: str, body: bytes) -> None:
    record = writer.create_warc_record(
        url,
        "response",
        payload=BytesIO(body),
        http_headers=StatusAndHeaders(
            "200 OK",
            [("Content-Type", "text/html; charset=utf-8")],
            protocol="HTTP/1.1",
        ),
        warc_headers_dict={"WARC-Identified-Payload-Type": "text/html"},
    )
    writer.write_record(record)


def _write_fixture_warc(path: Path) -> None:
    with path.open("wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        writer.write_record(writer.create_warcinfo_record(path.name, {"software": "test"}))
        _write_response(
            writer,
            url="https://example.com/one",
            body=b"<html><nav>raw navigation survives</nav></html> " * 8,
        )
        _write_response(
            writer,
            url="https://example.com/two",
            body=b"<script>var noisy = 42;</script> unextracted payload " * 8,
        )


def _write_materialize_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    warc_path = source_dir / "fixture.warc.gz"
    _write_fixture_warc(warc_path)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "0_raw_cc",
                "crawl_id": "CC-MAIN-TEST",
                "policy_revision": "0_raw_cc-test-v1",
                "downloads": [
                    {
                        "local_path": "/different-machine/source/fixture.warc.gz",
                        "bytes": warc_path.stat().st_size,
                        "sha256": _sha256(warc_path),
                    }
                ],
            }
        )
    )
    tokenizer_dir = Path("assets/hf/Mistral-7B-v0.1").resolve()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[stage]
id = "0_raw_cc"
crawl_id = "CC-MAIN-TEST"
policy_revision = "0_raw_cc-test-v1"

[probe]
input_path = "{warc_path}"

[census]
output_dir = "{tmp_path / 'census'}"
tokenizer_path = "{tokenizer_dir / 'tokenizer.json'}"
tokenizer_batch_size = 2
text_mime_prefixes = ["text/"]
text_mime_types = []
text_mime_suffixes = []

[materialize]
inventory_manifest = "{inventory_path}"
source_dir = "{source_dir}"
work_dir = "{tmp_path / 'work'}"
output_root = "{tmp_path / 'output'}"
sequence_length = 7
samples_per_shard = 3
shuffle_seed = 42
""".strip()
    )
    return config_path, warc_path


def test_materializer_uses_frozen_reader_adds_eos_and_resumes_exactly(tmp_path: Path) -> None:
    config_path, _ = _write_materialize_fixture(tmp_path)
    config = load_materialize_config(config_path)

    first = run_materialization(config)
    tokenized_path = next(config.work_dir.resolve().glob("*/**/*.ds"))
    original_mtime = tokenized_path.stat().st_mtime_ns
    second = run_materialization(config)

    assert second == first
    assert tokenized_path.stat().st_mtime_ns == original_mtime
    assert first["policy_revision"] == "0_raw_cc-test-v1"
    assert first["counts"]["source_documents"] == 2
    assert first["counts"]["materialized_tokens"] % 8 == 0
    assert first["counts"]["discarded_final_tokens"] < 8
    output_dir = Path(first["dataset_dir"])
    tokens = [token for shard in sorted(output_dir.glob("*.ds")) for token in _read_tokens(shard)]
    assert tokens.count(2) == first["counts"]["document_boundaries"]
    decoded = Tokenizer.from_file(str(config.tokenizer_path)).decode(tokens)
    assert "<html>" in decoded
    assert "raw navigation survives" in decoded
    assert first["tokenizer"]["eos_id"] == 2

    census = run_census(load_census_config(config_path))
    assert census["counts"]["training_tokens_with_eos"] == first["counts"][
        "source_tokens"
    ]

    view_path = tmp_path / "training-view.json"
    create_training_view(
        dataset_dir=output_dir,
        output_path=view_path,
        sequence_length=7,
        global_batch_size=1,
        training_steps=1,
        seed=42,
    )
    dataset = DataTroveTrainingDataset(view_path, verify_hashes=True)
    inputs, labels = dataset[0]
    assert inputs["input"].shape == labels.shape == (7,)


def test_materializer_rejects_inventory_source_corruption(tmp_path: Path) -> None:
    config_path, warc_path = _write_materialize_fixture(tmp_path)
    config = load_materialize_config(config_path)
    warc_path.write_bytes(warc_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="size mismatch"):
        run_materialization(config)


def test_materializer_rejects_corrupted_completed_token_artifact(tmp_path: Path) -> None:
    config_path, _ = _write_materialize_fixture(tmp_path)
    config = load_materialize_config(config_path)
    run_materialization(config)
    tokenized_path = next(config.work_dir.resolve().glob("*/**/*.ds"))
    tokenized_path.write_bytes(tokenized_path.read_bytes() + b"\x00\x00")

    with pytest.raises(ValueError, match="artifact size mismatch"):
        run_materialization(config)


def test_materialize_cli_supports_bounded_inventory_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = _write_materialize_fixture(tmp_path)

    main(["materialize", "--config", str(config_path), "--limit", "1"])

    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["source_warcs"] == 1
    assert Path(result["dataset_dir"]).is_dir()
