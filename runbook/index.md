# Web Curation Lab Runbook Index

This runbook is the shared memory for humans and agents working on Web Curation
Lab. It records active investigations, reusable workflows, run artifacts, and
current conclusions. Keep it factual and command-oriented.

## Rules

- Read this index before non-trivial project work.
- Update the relevant stream when work changes project state, run status,
  conclusions, or the next action.
- Use actor/date headers: `## YYYY-MM-DD [codex] short title`.
- Prefer exact config names, local artifact paths, commit hashes, and commands.
- Mark estimates as estimates; preserve measurement evidence for factual claims.
- Do not write secrets, cloud IPs or hostnames, SSH details, tokens, or private
  provider metadata.
- If a stream is complete, move it to `archive/YYYY/` and update the archive
  index.

## Current Posture (2026-08-14)

- **Framework**: TorchTitan, vendored at upstream revision
  `95007d2175febe9607a5aac7a37d2b981dad55fd`.
- **Architecture**: wide dense Qwen3 is the current choice: 10 layers, width
  1,024, 153.38M parameters. It won the initial end-to-end speed tradeoff; the
  cloud artifacts still need to be copied into the repository's benchmark
  evidence format.
- **Tokenizer**: pinned 32,000-token `mistralai/Mistral-7B-v0.1` tokenizer.
- **Training scale**: target experiment starts at up to 100B retained tokens on
  an 8xH100 node; runtime remains an estimate until production batch size and
  steady-state throughput are measured on that topology.
- **Data**: DataTrove v0.9.0 is vendored; bounded inspection, full one-WARC
  census, decision-audit, and resumable training-shard materialization paths for
  `0_raw_cc` are implemented and locally
  validated. The finalized one-file census measured 1.630B training tokens,
  and the deterministic ten-WARC pilot measured a 1.598B-token mean with 1.58%
  coefficient of variation. The resulting estimate is 63 WARCs for 100B; the
  exact inventory and packed-token cap are not yet frozen. A real one-WARC
  materialization pilot now passes source integrity, EOS, exact document-index,
  sample alignment, hash, and training-loader checks. It also corrected an old
  BOS-inclusive census overcount of one token per document. The audit supports the current MIME
  routing. Mechanical policy revision `0_raw_cc-v1` now uses a documented
  best-effort replacement fallback after strict decoding candidates fail. The
  provisional schedule uses a fixed Common Crawl
  WARC inventory: `0_raw_cc` is a 100B-token raw
  textual-response-payload model, S1 performs extraction only, then nested
  language/hygiene, deduplication, heuristic quality, and learned
  quality/diversity stages follow at their natural retained budgets. The exact
  crawl and processing contract remain subject to correctness and scale pilots.
- **Evaluation**: Paloma statistics, pinned OLMo Eval orchestration, authenticated
  frozen assets, packaging, and unit contracts are implemented. GPU parity,
  distributed Paloma accounting, and the limited 83-task general smoke pass.
  Full suites are reserved for milestone models.
- **Checkpoint export**: resumable DCP save/load, native FP32 HF export, exact
  config/tokenizer packaging, file hashes, Transformers parity, and BF16
  inference all pass on GPU. Production scored configs export BF16.

## Active Streams

- [Training and architecture](streams/training_and_architecture.md)
- [Data curation](streams/data_curation.md)
- [Evaluation](streams/evaluation.md)
- [Cloud runs](streams/cloud_runs.md)

## References

- [Cloud training](references/cloud_training.md)
- [Hugging Face export](references/huggingface_export.md)

## Archive

- [2026 archive index](archive/2026/index.md)
