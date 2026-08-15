#!/usr/bin/env bash

# Source this file for copy/paste command templates. Replace placeholders before
# running. Do not store secrets or host-specific credentials here.

set -euo pipefail

web_curation_check() {
  uv run pytest
  uv run ruff check .
  git diff --check
}

web_curation_download_tokenizer() {
  uv run python scripts/download_tokenizer.py
}

web_curation_train_smoke() {
  local config_name="${1:?TorchTitan config name required}"
  local gpu_count="${2:-1}"

  uv run torchrun \
    --standalone \
    --nproc-per-node="${gpu_count}" \
    -m torchtitan.train \
    --module web_curation_lab.training \
    --config "${config_name}" \
    --training.steps 20
}

web_curation_train_8gpu() {
  local config_name="${1:?TorchTitan config name required}"

  uv run torchrun \
    --standalone \
    --nproc-per-node=8 \
    -m torchtitan.train \
    --module web_curation_lab.training \
    --config "${config_name}"
}
