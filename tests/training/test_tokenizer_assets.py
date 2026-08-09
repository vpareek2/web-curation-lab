from pathlib import Path

import pytest
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from web_curation_lab.tokenizer_assets import (
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_VOCAB_SIZE,
    TOKENIZER_PATH,
    verify_tokenizer,
)


def test_pinned_tokenizer_assets() -> None:
    if not Path(TOKENIZER_PATH).is_dir():
        pytest.skip("Run `uv run python scripts/download_tokenizer.py` first")
    verify_tokenizer()

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_PATH))
    token_ids = tokenizer.encode("hello web", add_bos=True, add_eos=True)
    assert tokenizer.get_vocab_size() == EXPECTED_VOCAB_SIZE
    assert tokenizer.bos_id == EXPECTED_BOS_ID
    assert tokenizer.eos_id == EXPECTED_EOS_ID
    assert token_ids[0] == EXPECTED_BOS_ID
    assert token_ids[-1] == EXPECTED_EOS_ID
