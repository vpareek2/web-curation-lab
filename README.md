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
uv sync --group dev
uv run pytest
```

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
