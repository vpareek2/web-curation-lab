"""Pinned tokenizer manifest and verification helpers."""

from __future__ import annotations

import hashlib
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
TOKENIZER_SHA256 = {
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "tokenizer.json": "11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720",
    "tokenizer.model": "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055",
    "tokenizer_config.json": "ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99",
}
EXPECTED_VOCAB_SIZE = 32_000
EXPECTED_BOS_ID = 1
EXPECTED_EOS_ID = 2


def verify_tokenizer(path: Path = TOKENIZER_PATH) -> None:
    """Verify the downloaded files and the invariants used by model configs."""

    missing = [name for name in TOKENIZER_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tokenizer files in {path}: {missing}")
    for name, expected in TOKENIZER_SHA256.items():
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Tokenizer SHA-256 mismatch for {path / name}")

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
