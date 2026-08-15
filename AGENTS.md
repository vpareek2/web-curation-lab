# Web Curation Lab: What We're Building

We are building a small, reproducible language-model experiment that measures
how progressive web-data curation changes model quality, token efficiency, and
training cost. This is an informal, technically honest blog project rather than
a publication-scale benchmark suite.

The working experiment starts from a realistically large source pool and
filters it down through nested stages. Do not assume that each stage can draw a
fresh random sample of arbitrary size. Corpus availability, retention, and
pipeline throughput are experimental results.

TorchTitan is the training framework. The current selected candidate is the
wide dense Qwen3 model: 10 layers, width 1,024, 8 query heads, 2 KV heads,
3,072-wide SwiGLU, 32,000-token vocabulary, and 153.38M parameters. Treat that
choice as current project state, not an eternal constraint; changing it requires
new measured evidence.

## Public Repo Contract

- Do not commit secrets, cloud IPs or hostnames, SSH details, tokens, provider
  account metadata, or internal-only paths.
- The supported local environment and command path uses `uv`.
- Keep project code outside `third_party/torchtitan`; avoid vendor changes so
  upstream subtree updates remain tractable.
- Local run artifacts are canonical. Dashboards may mirror them but are not a
  substitute for logs, configs, checkpoints, profiles, and summaries.
- Do not claim throughput, MFU, loss, retention, or evaluation results without
  preserved artifact evidence.

## Execution Environment

- Assume CUDA, PyTorch, kernel, and hardware behavior is version-sensitive.
- Verify GPU model/count, memory, driver/runtime, installed kernels, tokenizer
  assets, and prepared data before making environment claims.
- Do not introduce another dependency manager or installation path. Use `uv`.
- The model fits on each target GPU; replicated data parallelism is the default
  topology unless measurements justify something more complex.

## Agent Protocol

### Session Start (Always)

- Read `AGENTS.md`, then `AGENTS.md.local` if present.
- Read `runbook/index.md` before non-trivial project work. If the task maps to
  an active stream or reference, read that file too.
- For execution tasks, inspect the current branch and working tree before
  editing.

### Mode Gates

**No-Edits Mode**

- Triggered when the user asks for review, discussion, planning, or explicitly
  says not to make changes.
- Do not modify tracked files, install dependencies, run destructive git
  operations, or change GitHub/cloud state.
- Exit only when the user explicitly authorizes execution.

**Execution Mode**

- Default when the user asks to implement, fix, build, or run something.
- Keep execution inside the authorized scope and verify the result.

### Scope Lock

- Before editing, identify the exact files that need to change.
- Preserve unrelated user changes and do not rewrite stable code without need.
- For ports and refactors, preserve semantics unless the user approves a
  behavioral change.
- After fixing a bug pattern, search for matching occurrences in scope.

### Do Not Guess

- Verify configuration values, run status, file contents, environment state,
  model dimensions, token counts, and results from code or artifacts.
- Clearly distinguish measurements from estimates and pending hypotheses.
- Never turn an absent artifact into a remembered benchmark number.

### Git Safety

- Do not run `git checkout`, `git restore`, `git reset`, or `git clean` without
  explicit approval and an explanation of what could be lost.
- Never push or commit unless explicitly asked.
- Preserve the user's current branch and unrelated working-tree changes.

### Build and Run Discipline

- Launch expensive data processing or cloud training only when explicitly
  requested and when preflight requirements are satisfied.
- Run narrow tests for the touched surface; run broader tests when changing
  shared training, data manifests, evaluation, checkpointing, or logging.
- For performance comparisons, keep model, tokens, batch, compile behavior, and
  measurement windows comparable, and record intentional differences.
- The learning-rate plan is WSD expressed as fractions of each run. Current
  benchmark defaults use 10% warmup and 15% cooldown, but the final percentages
  are not frozen until explicitly recorded as a decision.

### Output Completeness

- When asked for full commands, provide copy/paste-ready commands including
  `cd`, config names, paths, and flags.
- Do not truncate requested logs or diffs.

### Runbook Protocol

- The tracked `runbook/` directory is shared project memory, not private
  scratch space.
- Update the relevant `runbook/streams/*.md` when work changes run status,
  validation results, artifact locations, known constraints, decisions, or next
  actions.
- Use headers like `## YYYY-MM-DD [codex] short title`.
- Keep entries terse and factual. Prefer bullets and fenced commands.
- Every substantive entry should include:
  - `Context`: what changed or was investigated.
  - `Commands`: exact commands run, including `cd`, config paths, and flags.
  - `Artifacts`: config, output, manifest, checkpoint, profile, evaluation, or
    commit paths/identifiers.
  - `Result`: observed status, measurements, or exact error. Do not write only
    “works.”
  - `Next`: the next action or why none remains.
- If no command ran, say so and name the source of the conclusion (for example,
  code inspection, artifact inspection, or a user decision).
- When logging a failure, include the failing command, exact error line,
  suspected cause, and remediation plan.
- Add new workstreams to `runbook/index.md`. Archive a stream under
  `runbook/archive/YYYY/` only when it is genuinely complete, then update the
  archive index.
- Never record secrets, cloud IPs or hostnames, SSH keys, tokens, or private
  provider details.

## Research Workflow

1. **Freeze the training contract**: architecture, tokenizer, optimizer, WSD
   fractions, precision, sequence length, and batch policy.
2. **Build evaluation first**: establish a local smoke path and a reproducible
   full evaluation path before expensive training.
3. **Pilot the data funnel**: run a manageable shard through every stage,
   measure retention and I/O, audit samples, then freeze thresholds.
4. **Process nested stages**: keep stable document IDs, filter reasons, scores,
   duplicate clusters, and per-stage manifests. Avoid copying canonical text
   unnecessarily.
5. **Train every stage**: preserve initialization and training settings, train
   over all tokens retained by that stage, and checkpoint at meaningful token
   counts.
6. **Evaluate and report**: compare quality against both tokens and measured
   accelerator time, including domain regressions and accepted/rejected samples.

## Project Principles

- One clear path for each supported operation.
- Explicit configuration and artifact provenance over hidden behavior.
- Stable document IDs and nested manifests make curation auditable.
- Data retention is measured, not forced to a predetermined quota.
- Profile end-to-end throughput; model-only FLOPs are insufficient.
- Fail loudly when tokenizer, manifest, checkpoint, or architecture metadata do
  not match.
- Prefer existing, proven data/evaluation infrastructure over rebuilding WARC
  parsing, distributed MinHash, or evaluation harnesses from scratch.
- Keep the blog narrative accessible while preserving enough evidence for
  another person to reproduce the experiment.

## Important Paths

- `src/web_curation_lab/training/`: TorchTitan configs and model definitions.
- `src/web_curation_lab/pipeline/`: data curation and manifest code.
- `src/web_curation_lab/evaluation/`: evaluation orchestration and reports.
- `configs/`: versioned data, model, training, and evaluation configuration.
- `benchmarks/training/`: architecture throughput evidence.
- `assets/hf/`: local tokenizer/model assets; generated and ignored.
- `outputs/`: local training artifacts; generated and ignored.
- `runbook/`: shared operational memory and reusable procedures.
- `third_party/torchtitan/`: vendored upstream TorchTitan source.

## Common Commands

```bash
uv sync --group dev --extra hybrid
uv run python scripts/download_tokenizer.py
uv run pytest

uv run torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide
```

Our ethos: make a compact, legible experiment whose conclusions come from
preserved evidence rather than research folklore.
