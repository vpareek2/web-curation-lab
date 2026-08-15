from __future__ import annotations

import gzip
from pathlib import Path

import pytest

pytest.importorskip("datatrove")

from web_curation_lab.pipeline.stages.raw_cc.pilot import (
    PilotConfig,
    _combined_summary,
    _selection_manifest,
    evenly_spaced_indices,
    load_pilot_config,
)


def _write_config(path: Path, listing: Path, source: Path, output: Path) -> None:
    path.write_text(
        f"""
[stage]
id = "0_raw_cc"
crawl_id = "CC-MAIN-TEST"
policy_revision = "0_raw_cc-test-v1"

[probe]
input_path = "input.warc.gz"

[census]
output_dir = "census"
tokenizer_path = "tokenizer/tokenizer.json"
tokenizer_batch_size = 1
text_mime_prefixes = ["text/"]
text_mime_types = []
text_mime_suffixes = []

[pilot]
listing_path = "{listing}"
source_dir = "{source}"
output_dir = "{output}"
base_url = "https://data.commoncrawl.org/"
sample_size = 3
expected_listing_entries = 5
target_training_tokens = 100000000000
download_chunk_bytes = 1024
download_timeout_seconds = 10
download_retries = 2
""".strip()
    )


def test_evenly_spaced_indices_cover_listing_endpoints() -> None:
    assert evenly_spaced_indices(100_000, 10) == [
        0,
        11_111,
        22_222,
        33_333,
        44_444,
        55_555,
        66_666,
        77_777,
        88_888,
        99_999,
    ]


def test_pilot_selection_is_reproducible_and_records_listing_hash(tmp_path: Path) -> None:
    listing = tmp_path / "warc.paths.gz"
    paths = [
        f"crawl-data/CC-MAIN-TEST/segments/{index}/warc/file-{index}.warc.gz"
        for index in range(5)
    ]
    with gzip.open(listing, "wt") as handle:
        handle.write("\n".join(paths) + "\n")
    config_path = tmp_path / "config.toml"
    _write_config(config_path, listing, tmp_path / "source", tmp_path / "output")

    config, census = load_pilot_config(config_path)
    manifest = _selection_manifest(config, paths)

    assert config.base_url == "https://data.commoncrawl.org"
    assert census.policy_revision == "0_raw_cc-test-v1"
    assert [row["listing_line_one_based"] for row in manifest["selected"]] == [1, 3, 5]
    assert [row["common_crawl_path"] for row in manifest["selected"]] == [
        paths[0],
        paths[2],
        paths[4],
    ]
    assert len(manifest["listing"]["sha256"]) == 64


def test_pilot_config_rejects_sample_larger_than_listing(tmp_path: Path) -> None:
    listing = tmp_path / "warc.paths.gz"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, listing, tmp_path / "source", tmp_path / "output")
    text = config_path.read_text().replace("sample_size = 3", "sample_size = 6")
    config_path.write_text(text)

    with pytest.raises(ValueError, match="must cover the sample"):
        load_pilot_config(config_path)


def test_combined_summary_derives_target_estimate_from_sample_mean(tmp_path: Path) -> None:
    config = PilotConfig(
        stage_id="0_raw_cc",
        crawl_id="CC-MAIN-TEST",
        policy_revision="0_raw_cc-test-v1",
        listing_path=tmp_path / "listing.gz",
        source_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        base_url="https://data.commoncrawl.org",
        sample_size=2,
        expected_listing_entries=2,
        target_training_tokens=100,
        download_chunk_bytes=1,
        download_timeout_seconds=1,
        download_retries=1,
    )
    selection = {
        "selection_algorithm": "test",
        "selected": [
            {"listing_line_one_based": 1, "common_crawl_path": "a.warc.gz"},
            {"listing_line_one_based": 2, "common_crawl_path": "b.warc.gz"},
        ],
    }
    runs = [
        {
            "_summary_path": f"{name}.json",
            "input": {"bytes": compressed_bytes},
            "counts": {
                "training_tokens_with_eos": tokens,
                "decoded_documents": documents,
                "response_records": documents + 2,
                "decoded_with_replacement": 1,
                "dropped_non_text_mime": 1,
                "dropped_empty_text": 1,
            },
        }
        for name, compressed_bytes, tokens, documents in (
            ("a", 10, 20, 8),
            ("b", 20, 30, 9),
        )
    ]

    summary = _combined_summary(config, selection, runs)

    assert summary["sample_totals"] == {
        "training_tokens_with_eos": 50,
        "decoded_documents": 17,
        "compressed_bytes": 30,
    }
    assert summary["distributions"]["training_tokens_with_eos_per_warc"]["mean"] == 25
    assert summary["estimate"]["warc_files_for_target_at_sample_mean"] == 4
