from pathlib import Path

import pytest

from web_curation_eval import assets
from web_curation_eval.provenance import sha256_file, verify_inventory


def test_mismatched_manifest_fails(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual")
    inventory = [
        {
            "path": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": "0" * 64,
        }
    ]

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_inventory(tmp_path, inventory)

    inventory[0]["sha256"] = sha256_file(artifact)
    verify_inventory(tmp_path, inventory)


def test_inaccessible_paloma_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "evaluation.toml"
    config.write_text(
        f'''[assets]
directory = "{tmp_path / "assets"}"
[health]
split = "val"
target_tokens = 10
output_file = "{tmp_path / "health.jsonl"}"
tokenizer_path = "missing-tokenizer"
[statistics]
dataset_repo = "allenai/paloma"
dataset_revision = "PALOMA_REVISION_REQUIRED"
[general]
tasks = ["olmobase:easy:qa:rc"]
'''
    )

    def deny(*_args, **_kwargs):
        raise PermissionError("gated")

    monkeypatch.setattr(assets.HfApi, "dataset_info", deny)
    with pytest.raises(RuntimeError, match="Accept the AI2 license"):
        assets.prepare_assets(config)


def test_general_environment_forwards_cached_hub_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assets, "get_token", lambda: "fixture-token")

    environment = assets.general_environment(tmp_path / "isolated-cache")

    assert environment["HF_HOME"] == str(tmp_path / "isolated-cache")
    assert environment["HF_TOKEN"] == "fixture-token"


@pytest.mark.parametrize(
    ("current", "document", "target", "expected"),
    [
        (0, 500, 100, True),
        (40, 30, 100, True),
        (70, 50, 100, True),
        (70, 80, 100, False),
    ],
)
def test_stratified_holdout_chooses_closest_document_boundary(
    current: int, document: int, target: int, expected: bool
) -> None:
    assert (
        assets.include_for_stratified_target(
            current_tokens=current,
            document_tokens=document,
            target_tokens=target,
        )
        is expected
    )
