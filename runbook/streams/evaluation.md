# Evaluation

Purpose: track evaluation task selection, harness integration, held-out loss,
decontamination, checkpoint evaluation, reporting, and regression evidence.

## 2026-08-12 [codex] Evaluation is the next local-first build target

Context:

- Evaluation currently contains only a package scaffold.
- Evaluation should exist before expensive data processing and training so the
  project does not discover checkpoint-format or task incompatibilities after
  a long run.
- The proposal names domain-sensitive held-out loss (Paloma) and DCLM CORE as
  likely primary/secondary views, with individual small-model tasks useful for
  interpretation. Exact suite versions and score aggregation still need to be
  verified and frozen.

Commands:

- No evaluation command was run for this entry.
- Source: repository inspection and `project.md`.

Artifacts:

- Evaluation package: `src/web_curation_lab/evaluation/`
- Future evaluation configs: `configs/evaluation/`
- Export dependency: `runbook/references/huggingface_export.md`

Result:

- No evaluation harness, task configuration, baseline result, or report schema
  is implemented yet.
- Most orchestration and smoke testing can be developed locally with a tiny or
  public HF checkpoint. Full-suite timing and the project's trained
  checkpoints will require appropriate compute later.

Next:

- Verify the current DCLM CORE task definition/version and Paloma integration.
- Choose a tiny local smoke subset plus the full scored suite.
- Define a versioned result schema containing checkpoint identity, task/version,
  tokenizer/config identity, command, raw metrics, and aggregate score.
- Test the entire HF-export-to-evaluation path before the first long run.

## 2026-08-12 [codex] Paloma and modern general suite implemented

Context:

- Implemented the two first-class suites around packaged HF checkpoints.
- Investigated DCLM CORE, lm-evaluation-harness, LightEval, OLMES, and the new
  OLMo Eval workbench. Exact DCLM CORE requires its old OpenLM/LLM Foundry task
  path; running similar tasks elsewhere would not be numerically equivalent.
- Selected OLMo Eval's `olmobase:easy:qa:rc` suite at commit
  `479f90ea137a2a044a33aa7c81128d400bd027c4`. It is current, base-model
  oriented, ships a frozen uv lock, and records requests/predictions. The pinned
  suite expands to 83 task configurations. Results are explicitly not labeled
  DCLM `Core_v2`.

Commands:

```bash
cd web-curation-lab
uv lock --project evaluation
uv run --project evaluation --frozen pytest -q evaluation/tests
uv run --project evaluation --frozen ruff check \
  evaluation/src evaluation/tests evaluation/scripts
uv run pytest -q
```

Artifacts:

- Environment: `evaluation/pyproject.toml`, `evaluation/uv.lock`
- Configs: `configs/evaluation/health.toml`,
  `configs/evaluation/full.toml`, `configs/evaluation/general.toml`
- Statistics evaluator: `evaluation/src/web_curation_eval/statistics.py`
- General wrapper: `evaluation/src/web_curation_eval/general.py`
- Asset preparation: `evaluation/src/web_curation_eval/assets.py`,
  `evaluation/scripts/prepare_general_assets.py`
- Pinned framework: `third_party/olmo-eval/`

Result:

- Paloma evaluation preserves document boundaries, prepends but does not score
  BOS, does not append EOS, and splits long documents into disjoint 2,048-token
  segments. It emits exact sums and derived loss, perplexity, bits/token, and
  bits/byte at micro, source, and domain levels.
- Asset preparation resolves and hashes the gated Paloma snapshot, creates the
  deterministic source-stratified 2M-token validation holdout, expands every
  general-suite task, records dataset revisions/configs/counts, and hashes the
  frozen HF cache. General evaluation runs against that cache in offline mode.
- OLMo Eval uses one model replica per GPU rather than tensor parallelism for
  this 153M model. Any task error or missing/non-numeric suite summary fails the
  run.
- The pinned CLI accepted the exact wrapper flags in a local `--dry-run` and
  expanded `olmobase:easy:qa:rc` to 83 task configurations. This is a broad
  suite despite its `easy` capability label, so the first complete GPU run must
  record wall time before milestone costs are finalized.
- Evaluation tests: 13 passed. Root tests: 12 passed, 11 skipped on macOS;
  CUDA-only coverage remains pending.
- DCLM and OLMES candidate trees created during investigation were moved to
  `/private/tmp/web-curation-dclm-replaced` and
  `/private/tmp/web-curation-olmes-replaced`; they are not project artifacts.

Next:

- Accept/authenticate the Paloma license, then run `prepare-assets` before data
  curation so one decontamination mask can cover all frozen eval sources.
- On the GPU node: run HF parity, TorchTitan health validation, a limited
  single-/multi-GPU Paloma comparison, a general-suite five-instance smoke,
  then one complete general evaluation. Record measured runtimes here.

## 2026-08-13 [codex] Asset preparation fixed after first authenticated run

Context:

- Paloma downloaded successfully, including all 1,145 snapshot files, and the
  2M-token health holdout was written.
- General-suite preparation then treated OLMo Eval's embedded fixed few-shot
  alias `olmes_arc_challenge_fixed` as a Hugging Face repository and received a
  401. Setting an isolated `HF_HOME` for the frozen general cache also hid the
  token saved in the user's normal Hugging Face cache.
- The resumed run exposed a second provenance-only edge case: the Basic Skills
  tasks use the generic `json` loader with files from `allenai/basic-skills`, so
  the recorder initially queried a nonexistent Hub repository named `json`.

Result:

- Fixed few-shot aliases ending in `_fixed` are now recorded as embedded task
  assets and are not queried on the Hub.
- Generic JSON/CSV/Parquet/text sources now resolve the actual Hub repository
  from their `hf://datasets/...` data-file URL.
- The resolved Hugging Face token is forwarded to the isolated subprocess
  without being written to logs or manifests.
- Asset preparation completed at Paloma revision
  `65cd6fc59dba021b21db414fa5e8d7765ffbe5e6`: 83 general task records, 20
  resolved dataset revisions, and a document-intact, all-source health holdout
  of 2,046,998 tokens (2.35% over its 2M target).
- General cache paths are relative so the 659MB asset directory is portable to
  the GPU checkout. Reruns resume from the existing caches.

## 2026-08-13 [codex] GPU evaluation preflight passed

Context:

- Used the native TorchTitan FP32 HF export from the two-H200, 12-step training
  preflight.
- OLMo Eval requires Python >=3.12. Without an explicit interpreter selection,
  uv selected Python 3.14 and pinned llvmlite could not build.
- Several official tasks read cached files through `hf://` URLs. Hugging Face's
  filesystem adapter requires a metadata lookup even when content blobs are
  cached, so strict offline mode could not prepare those tasks.

Commands:

```bash
uv run --extra eval-preflight web-curation-check-hf-parity \
  --model outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --output outputs/preflight/qwen3_150m_wide/hf_parity.json
uv run --project evaluation web-curation-eval statistics \
  --model outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --config configs/evaluation/health.toml \
  --output outputs/preflight/qwen3_150m_wide/statistics-1gpu.json
torchrun --standalone --nproc-per-node=2 -m web_curation_eval.cli statistics \
  --model outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --config configs/evaluation/health.toml \
  --output outputs/preflight/qwen3_150m_wide/statistics-2gpu.json
uv run --project evaluation web-curation-eval general \
  --model outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --config configs/evaluation/health.toml \
  --output outputs/preflight/qwen3_150m_wide/general-smoke --num-gpus 2
```

Result:

- TorchTitan and Transformers produced identical argmaxes. KL divergence was
  effectively zero (-1.97e-7 from numerical roundoff), below the 1e-6 limit;
  maximum absolute logit error was 1.91e-6. BF16 load/generation passed.
- One- and two-GPU Paloma runs exactly matched at 100 documents, 255,903
  predicted tokens, and 593,371 UTF-8 bytes. NLL differed by 0.00565 total, or
  2.21e-8 per token, due to reduction order. Wall time was 5.02s and 3.49s.
- The general smoke completed all 83 tasks/415 limited instances with no
  errors in 118.86s including vLLM startup. The smoke aggregate was 0.25771 but
  has no scientific meaning for a model trained for only 12 steps.
- The wrapper now pins OLMo Eval to Python 3.12, resolves uv in non-interactive
  environments, permits required Hub metadata calls, and verifies the frozen
  cache inventory both before and after evaluation to reject source drift.
- Compact evidence is retained under
  `benchmarks/training/h200_preflight_2026-08-13/`; checkpoint weights were not
  copied back.

Next:

- Run complete Paloma and OLMo Eval only for actual milestone checkpoints.
- A single requested aggregate suite currently uses one vLLM replica even when
  two GPUs are supplied; improve task-level partitioning before optimizing full
  general-suite wall time.
