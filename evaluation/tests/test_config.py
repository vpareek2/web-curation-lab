import json
from pathlib import Path

import pytest

from web_curation_eval.config import load_config


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "eval.toml"
    path.write_text(
        f'''[assets]
directory = "{tmp_path}/assets"
[health]
split = "val"
target_tokens = 2000000
output_file = "{tmp_path}/health.jsonl"
tokenizer_path = "assets/tokenizer"
[statistics]
dataset_repo = "allenai/paloma"
dataset_revision = "PALOMA_REVISION_REQUIRED"
split = "test"
max_sequence_length = 2048
batch_size = 4
dtype = "bfloat16"
[general]
framework_root = "third_party/olmo-eval"
tasks = ["olmobase:easy:qa:rc"]
max_length = 2048
'''
    )
    return path


def test_placeholder_resolves_from_asset_manifest(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "paloma_manifest.json").write_text(json.dumps({"revision": "abc123"}))

    assert load_config(config_path).statistics.dataset_revision == "abc123"


def test_unprepared_placeholder_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepare-assets"):
        load_config(_config(tmp_path))
