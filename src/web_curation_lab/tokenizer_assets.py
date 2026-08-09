"""Pinned tokenizer manifest and verification helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_REPO_ID = "mistralai/Mistral-7B-v0.1"
TOKENIZER_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
TOKENIZER_PATH = Path("assets/hf/Mistral-7B-v0.1")
TOKENIZER_FILES = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
EXPECTED_VOCAB_SIZE = 32_000
EXPECTED_BOS_ID = 1
EXPECTED_EOS_ID = 2


def verify_tokenizer(path: Path = TOKENIZER_PATH) -> None:
    """Verify the downloaded files and the invariants used by model configs."""

    missing = [name for name in TOKENIZER_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tokenizer files in {path}: {missing}")

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    bos_token = tokenizer_config["bos_token"]
    eos_token = tokenizer_config["eos_token"]

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    bos_id = tokenizer.token_to_id(bos_token)
    eos_id = tokenizer.token_to_id(eos_token)
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected {EXPECTED_VOCAB_SIZE} tokens, found {vocab_size}")
    if bos_id != EXPECTED_BOS_ID or eos_id != EXPECTED_EOS_ID:
        raise ValueError(
            f"Expected BOS/EOS IDs {(EXPECTED_BOS_ID, EXPECTED_EOS_ID)}, "
            f"found {(bos_id, eos_id)}"
        )

    sample = "A small web corpus with punctuation, numbers 123, and Unicode: café."
    encoded = tokenizer.encode(sample).ids
    decoded = tokenizer.decode(encoded)
    if not encoded or not decoded.strip():
        raise ValueError("Tokenizer encode/decode smoke check returned empty output")
    if min(encoded) < 0 or max(encoded) >= EXPECTED_VOCAB_SIZE:
        raise ValueError("Tokenizer emitted an ID outside the configured vocabulary")

    print(
        f"Verified {TOKENIZER_REPO_ID}@{TOKENIZER_REVISION}: "
        f"vocab={vocab_size}, bos={bos_id}, eos={eos_id}, path={path}"
    )
