# Decision 0001: Use TorchTitan as the training framework

## Status

Accepted.

## Context

The project will train a roughly 100M--250M parameter decoder-only language
model on one eight-H100 node. The model fits on each GPU, so the expected
starting topology is replicated data parallelism. High-throughput kernels,
compilation, efficient input loading, checkpointing, and observability matter
more than large-model sharding features.

## Decision

Use TorchTitan as the training engine. Vendor upstream with Git subtree under
`third_party/torchtitan` and keep project-specific model, data, and experiment
code outside the vendored directory.

The initial upstream revision is:

`95007d2175febe9607a5aac7a37d2b981dad55fd`

## Consequences

- The complete project can be cloned as one repository.
- Upstream changes can be imported explicitly with `git subtree pull`.
- Vendor modifications should be avoided so upstream updates remain tractable.
- The CUDA and PyTorch versions remain intentionally unfrozen until H100
  throughput validation establishes a known-good environment.
