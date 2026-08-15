# Cloud Training Reference

Use this workflow for GPU validation and training. Never commit cloud IPs,
hostnames, SSH details, keys, tokens, or provider metadata.

## Setup

```bash
git clone https://github.com/vpareek2/web-curation-lab.git
cd web-curation-lab
uv sync --group dev --extra hybrid
uv run python scripts/download_tokenizer.py
uv run pytest
```

Use the environment/kernel commands validated for the target CUDA and PyTorch
versions. Once the 8xH100 preflight passes, record exact package versions here
without recording host-specific credentials.

## Device Verification

```bash
cd web-curation-lab
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, properties.total_memory)
PY
```

## Smoke Run

```bash
cd web-curation-lab
uv run torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide \
  --training.steps 20
```

The benchmark config currently expects eight data-parallel ranks. If using a
different GPU count, update or override the parallelism and global-batch
contract consistently; changing only `--nproc-per-node` is not sufficient.

## Before a Long Run

- Record the commit and clean/dirty tree state.
- Record GPU model/count and interconnect topology.
- Record `uv lock`/installed versions for PyTorch, CUDA-facing kernels, and
  TorchTitan.
- Verify the tokenizer revision and file hashes.
- Verify the prepared-data manifest, token count, and sample decoding.
- Run enough steps to pass compilation warmup and measure steady state.
- Measure per-GPU peak memory and test larger local batches.
- Exercise checkpoint save, resume, and final HF export.
- Confirm output storage capacity and a copy-back/upload path.

## What To Record In Streams

- Commit hash and working-tree state.
- Config name plus every override.
- GPU type/count and relevant topology, without host identifiers.
- Environment versions.
- Dataset/manifest identity and tokenizer revision.
- Warmup-excluded tokens/s, MFU when comparable, loss, peak memory, and exact
  measurement window.
- Checkpoint and profile artifact paths.
- Exact failures and remediation.
