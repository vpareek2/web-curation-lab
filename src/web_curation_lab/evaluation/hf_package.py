"""Package a native TorchTitan HF save as a complete Transformers directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from web_curation_lab.tokenizer_assets import (
    EXPECTED_VOCAB_SIZE,
    TOKENIZER_FILES,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    verify_tokenizer,
)
from web_curation_lab.training.models.qwen3 import ARCHITECTURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_hf_config(architecture_name: str, tokenizer_path: Path) -> dict[str, Any]:
    """Build the standard Qwen3 Transformers config from project-owned dimensions."""

    architecture = ARCHITECTURES[architecture_name]
    verify_tokenizer(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    tokenizer_config = json.loads((tokenizer_path / "tokenizer_config.json").read_text())
    bos_id = tokenizer.token_to_id(tokenizer_config["bos_token"])
    eos_id = tokenizer.token_to_id(tokenizer_config["eos_token"])
    if tokenizer.get_vocab_size(with_added_tokens=True) != architecture.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model architecture")
    if bos_id is None or eos_id is None:
        raise ValueError("Could not resolve BOS/EOS IDs from tokenizer assets")
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "head_dim": architecture.head_dim,
        "hidden_act": "silu",
        "hidden_size": architecture.dim,
        "initializer_range": 0.02,
        "intermediate_size": architecture.ffn_hidden_dim,
        "max_position_embeddings": architecture.max_seq_len,
        "model_type": "qwen3",
        "num_attention_heads": architecture.num_heads,
        "num_hidden_layers": architecture.num_layers,
        "num_key_value_heads": architecture.num_kv_heads,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": architecture.vocab_size,
    }


def package_checkpoint(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
    architecture_name: str,
    run_id: str,
    stage: str,
    step: int,
    tokens_seen: int,
    source_revision: str | None = None,
) -> Path:
    """Copy safetensors and pinned assets, then write a provenance manifest."""

    weights = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to package into non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in weights:
        shutil.copy2(source, output_dir / source.name)
    index = checkpoint_dir / "model.safetensors.index.json"
    if index.is_file():
        shutil.copy2(index, output_dir / index.name)
    for name in TOKENIZER_FILES:
        shutil.copy2(tokenizer_path / name, output_dir / name)
    config = build_hf_config(architecture_name, tokenizer_path)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "evaluation_manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "step": step,
        "tokens_seen": tokens_seen,
        "architecture": architecture_name,
        "parameter_count": ARCHITECTURES[architecture_name].parameter_count,
        "source_revision": source_revision or _git_revision(),
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "files": TOKENIZER_SHA256,
        },
        "files": files,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("assets/hf/Mistral-7B-v0.1"))
    parser.add_argument("--architecture", default="qwen3_150m_wide", choices=sorted(ARCHITECTURES))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--tokens-seen", type=int, required=True)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    manifest = package_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        architecture_name=args.architecture,
        run_id=args.run_id,
        stage=args.stage,
        step=args.step,
        tokens_seen=args.tokens_seen,
        source_revision=args.source_revision,
    )
    print(manifest)


if __name__ == "__main__":
    main()
