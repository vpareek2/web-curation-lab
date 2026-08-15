import json
from pathlib import Path

import pytest

from web_curation_lab.evaluation.hf_package import build_hf_config, package_checkpoint
from web_curation_lab.tokenizer_assets import TOKENIZER_PATH, TOKENIZER_SHA256


def test_wide_qwen3_config_matches_registry() -> None:
    if not TOKENIZER_PATH.is_dir():
        pytest.skip("Run the tokenizer download first")
    config = build_hf_config("qwen3_150m_wide", TOKENIZER_PATH)

    assert config["model_type"] == "qwen3"
    assert config["vocab_size"] == 32_000
    assert config["hidden_size"] == 1_024
    assert config["intermediate_size"] == 3_072
    assert config["num_hidden_layers"] == 10
    assert config["num_attention_heads"] == 8
    assert config["num_key_value_heads"] == 2
    assert config["head_dim"] == 128
    assert config["tie_word_embeddings"] is True


def test_package_writes_hashed_manifest(tmp_path: Path) -> None:
    if not TOKENIZER_PATH.is_dir():
        pytest.skip("Run the tokenizer download first")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    output = tmp_path / "hf"

    manifest_path = package_checkpoint(
        checkpoint_dir=checkpoint,
        output_dir=output,
        tokenizer_path=TOKENIZER_PATH,
        architecture_name="qwen3_150m_wide",
        run_id="fixture",
        stage="s0",
        step=10,
        tokens_seen=1_000,
        source_revision="deadbeef",
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["source_revision"] == "deadbeef"
    assert manifest["tokens_seen"] == 1_000
    assert manifest["tokenizer"]["files"] == TOKENIZER_SHA256
    assert {entry["path"] for entry in manifest["files"]} >= {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
