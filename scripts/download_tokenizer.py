"""Download and verify the tokenizer shared by all training configurations."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from web_curation_lab.tokenizer_assets import (
    TOKENIZER_FILES,
    TOKENIZER_PATH,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    verify_tokenizer,
)


def main() -> None:
    TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=TOKENIZER_REPO_ID,
        revision=TOKENIZER_REVISION,
        local_dir=TOKENIZER_PATH,
        allow_patterns=list(TOKENIZER_FILES),
    )
    verify_tokenizer()


if __name__ == "__main__":
    main()
