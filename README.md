# Web Curation Lab

An end-to-end experiment in progressively curating a large web corpus and
measuring how each stage changes the quality and training cost of a small
language model.

The repository is organized as a monorepo. TorchTitan is vendored under
`third_party/torchtitan`, while project-specific training, data, and evaluation
code lives in `src/web_curation_lab`.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environments,
dependency management, and command execution.

```bash
uv sync --group dev --extra hybrid
uv run python scripts/download_tokenizer.py
uv run pytest
```

All models use the 32,000-token tokenizer from
`mistralai/Mistral-7B-v0.1`, pinned to Hugging Face revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da`. The downloaded tokenizer files
live under `assets/hf/Mistral-7B-v0.1` and are not committed.

The exact PyTorch, CUDA, and TorchTitan environment for H100 training will be
locked after the initial single-node throughput validation. Until then, do not
upgrade the vendored TorchTitan revision casually.

## Repository map

- `src/web_curation_lab/training`: model recipes and TorchTitan integration.
- `src/web_curation_lab/pipeline`: corpus processing and curation stages.
- `src/web_curation_lab/evaluation`: evaluation orchestration and reporting.
- `configs`: versioned experiment configuration.
- `benchmarks/training`: framework and architecture throughput measurements.
- `infra`: container and cluster launch definitions.
- `docs`: design decisions and blog material.
- `third_party/torchtitan`: vendored upstream TorchTitan source.

See [`project.md`](project.md) for the initial project proposal.

## Reference training configuration

The initial architectures are:

- `curation_qwen3_150m_reference`: 16 layers, width 768, 148.86M parameters.
- `curation_qwen3_150m_wide`: 10 layers, width 1,024, 153.38M parameters.
- `curation_gpt_oss_dense_150m_reference`: 16 layers, width 768, 148.89M parameters.
- `curation_gpt_oss_dense_150m_wide`: 10 layers, width 1,024, 153.40M parameters.
- `curation_qwen3_5_150m_reference`: 16 layers, width 768, 150.09M parameters.
- `curation_qwen3_5_150m_wide`: 10 layers, width 1,024, 153.44M parameters.

The Qwen3.5 variants are text-only hybrid decoders: three GatedDeltaNet linear
attention blocks followed by one full-attention block. They use Qwen3.5-style
output-gated attention, partial RoPE, and 2x-width key/value mixer states. The
`varlen` attention backend is mandatory so recurrent and convolution state is
reset between packed documents.

The dense GPT-OSS variants alternate 128-token sliding-window and global
attention layers. They use TorchTitan's GPT-OSS attention implementation,
including learned per-head attention sinks and biased QKV/output projections,
but replace the routed experts with dense SwiGLU feed-forward blocks.

Each default run processes 500,170,752 tokens from the streaming C4 loader on
eight data-parallel GPUs.

Download the pinned tokenizer as shown above, then launch the configuration with:

```bash
uv run torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_reference
```

For a short cloud smoke test, append `--training.steps 20`.

The `hybrid` extra provides the Qwen3.5 variants' Flash Linear Attention
dependency and the prebuilt Hopper FlashAttention-3 kernel used by every
varlen-attention recipe. It is included in the development setup above; omit
it only when working exclusively with the dense Qwen3 recipes.
