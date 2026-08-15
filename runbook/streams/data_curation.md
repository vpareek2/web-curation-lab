# Data Curation

Purpose: track source-corpus selection, deterministic sampling, stable document
identity, filter stages, deduplication, token accounting, manifests, and data
pipeline performance.

## 2026-08-12 [codex] Current state before implementation

Context:

- Data-pipeline packages exist as scaffolding only; no production source,
  manifest schema, or filter implementation is committed.
- The experiment should reflect a realistic lab workflow: start from a very
  large available pool and progressively filter nested subsets. It should not
  assume the ability to obtain a fresh random 10B-token sample at every stage.
- The initial proposal discusses DCLM extraction-only data, but the base corpus
  and acquisition mechanics still need an explicit current decision.

Commands:

- No data command was run for this entry.
- Source: repository inspection, `project.md`, and user discussion.

Artifacts:

- Project proposal: `project.md`
- Pipeline package: `src/web_curation_lab/pipeline/`
- Future data configs: `configs/data/`

Result:

- No dataset has been downloaded, sampled, tokenized, or filtered in this
  checkout.
- Planned stages remain nested: permissive baseline, hygiene, deduplication,
  heuristic quality, and learned quality.
- Retained token counts are outcomes to measure, not quotas to force.

Next:

- Decide the exact base dataset, accessible shard inventory, licenses/terms,
  and deterministic selection strategy.
- Define a canonical document record and stable ID before writing filters.
- Build a small end-to-end pilot that records per-stage keep/drop reasons,
  token counts under the pinned tokenizer, throughput, temporary disk growth,
  and accepted/rejected samples.
- Freeze thresholds only after auditing the pilot.

## 2026-08-14 [codex] Provisional raw-Common-Crawl experiment schedule

Context:

- The intended blog story now begins before text extraction: acquire one fixed,
  deterministic Common Crawl WARC source inventory, train the first model on
  raw textual response payloads, then measure what successive processing stages
  retain and how the models change.
- DCLM is not the selected source corpus or processing framework. DataTrove is
  the provisional primary execution library; project code will own identities,
  manifests, stage semantics, audits, classifiers, and reporting.
- This is a high-level schedule subject to change after the correctness and
  scale pilots. Exact crawl, MIME/decoding policy, thresholds, dedup parameters,
  quality model, hardware, and retained-token budgets are not frozen.

Commands:

- No data-processing or training command was run.
- Source: user design decision and repository/runbook inspection.

Artifacts:

- Planned source inventory: `configs/data/` (not implemented)
- Planned pipeline code: `src/web_curation_lab/pipeline/`
- Planned local artifacts: `outputs/data/` (generated and ignored)

Result:

- The provisional model/data stages are:
  - **S0 — raw Common Crawl:** train for approximately 100B tokenizer tokens on
    textual HTTP response payloads from the fixed WARC inventory. Necessary
    container/transport decompression, deterministic charset decoding,
    text-capable MIME routing, evaluation decontamination, and mandatory
    legal/safety exclusions apply. Do not perform main-text extraction,
    language routing, normalization, deduplication, or quality filtering.
  - **S1 — extraction only:** run main-content extraction over the exact S0
    source records and train on all naturally retained extracted-text tokens;
    do not replenish the corpus to 100B.
  - **S2 — language and hygiene:** add target-language routing, structural
    validity, malformed/extreme-document, boilerplate, repetition, markup,
    symbol, and related inexpensive hygiene rules.
  - **S3 — deduplication:** add deterministic exact and fuzzy deduplication,
    preserving cluster IDs and representative-selection evidence.
  - **S4 — heuristic quality:** add a small, auditable collection of explicit
    web-quality rules rather than an opaque union of every published filter.
  - **S5 — learned quality and diversity:** add model-based quality selection
    with domain/diversity safeguards and train on its natural retained budget.
- S0 and S1 share record-level lineage but not identical text: each stable
  `document_id` maps to a raw payload and, when successful, an extracted-text
  representation. S1 onward should use nested document-level decisions.
- Every scored model uses the shared initialization and the same frozen
  training contract. Because WSD phases are percentage-based, each later
  stage's equal-token S0 comparison is an independent raw-payload run at that
  stage's retained-token budget, not a checkpoint from the 100B S0 run.
- The provisional execution schedule is:
  1. Freeze the crawl/source manifest, stable ID/schema, decoding/MIME contract,
     universal exclusions, and artifact schemas.
  2. Run a roughly 10-WARC-file correctness pilot through every stage.
  3. Run a deterministically distributed 50–100-file scale pilot; measure raw
     and extracted tokenizer yield, retention, throughput, memory, network, and
     temporary-storage amplification before provisioning the full source set.
  4. Freeze a source inventory that supplies the 100B-token S0 training stream,
     materialize S0, train it, and run milestone evaluation.
  5. Produce S1 from those exact source records, then train/evaluate S1 and its
     independent equal-token S0 baseline.
  6. Repeat the nested process for S2–S5, auditing accepted/rejected samples and
     recording natural retained documents/tokens at each stage.
  7. Report quality versus tokens and measured accelerator time alongside the
     complete data-retention and processing-cost funnel.
- The S0 hypothesis is that raw payload tokens are inefficient training data,
  but the outcome is not assumed; raw HTML/code structure may provide more
  signal than expected.

Next:

- Choose the Common Crawl snapshot and deterministic WARC-file selection rule.
- Define the exact `0_raw_cc` text-capable MIME, HTTP-status, decoding, document-boundary,
  decontamination, and mandatory-exclusion contract.
- Define the canonical record/manifest schema before implementing the pipeline.
- Specify pilot acceptance measurements and storage/hardware limits.

## 2026-08-14 [codex] Vendor DataTrove

Context:

- The first data stage is now named `0_raw_cc`; the earlier `S0` label is
  superseded for new code and documentation.
- DataTrove should follow the same vendored-upstream pattern as TorchTitan so
  its exact implementation remains reproducible and inspectable.

Commands:

```bash
cd /path/to/web-curation-lab
git remote add datatrove-upstream https://github.com/huggingface/datatrove.git
git fetch datatrove-upstream tag v0.9.0
git subtree add \
  --prefix=third_party/datatrove \
  datatrove-upstream v0.9.0 \
  -m "Vendor DataTrove v0.9.0"
```

Artifacts:

- Vendored source: `third_party/datatrove/`
- Upstream tag: `v0.9.0`
- Upstream commit: `87f7bad5c4a56ec648265fbf0b91d7d226bad428`
- Subtree import commit: `80e91d3676ccc72ed03aa1f0ce782bede502396f`

Result:

- DataTrove was imported as a Git subtree with the pinned upstream commit as
  the import commit's second parent.
- Existing uncommitted repository changes were preserved.
- DataTrove is not yet connected to the root uv environment, and no Common
  Crawl data has been downloaded or processed.

Next:

- Connect the vendored package through uv and prove a minimal local import.
- Then define the smallest `0_raw_cc` one-WARC correctness fixture.

## 2026-08-14 [codex] Connect DataTrove to uv

Context:

- The root environment needs an explicit, reproducible path to the vendored
  DataTrove source without installing data-processing dependencies for normal
  TorchTitan work.
- DataTrove 0.9.0 requires `huggingface-hub<1`, while the optional training
  kernel and evaluation-preflight environments require newer incompatible Hub
  versions. These extras are intentionally separate execution environments.

Commands:

```bash
cd web-curation-lab
uv lock
uv sync --extra data
uv run --extra data python -c "from importlib.metadata import version; import datatrove; from datatrove.pipeline.readers import WarcReader; print(version('datatrove'), datatrove.__path__[0], WarcReader.__module__)"
uv run --extra data pytest -q
```

Artifacts:

- Root dependency configuration: `pyproject.toml`
- Locked dependency graph: `uv.lock`
- Vendored package: `third_party/datatrove/`

Result:

- Added a `data` extra containing `datatrove[io]` and an editable uv source at
  `third_party/datatrove`.
- Declared `data` mutually exclusive with `hybrid` and `eval-preflight`, keeping
  their incompatible upstream Hugging Face Hub requirements explicit.
- Import verification resolved DataTrove 0.9.0 from the vendored source and
  imported `datatrove.pipeline.readers.warc.WarcReader` successfully.
- Root tests in the data environment passed: 12 passed and 12 macOS/Triton
  skips.

Next:

- Select one deterministic Common Crawl WARC file and implement the smallest
  `0_raw_cc` local correctness fixture.

## 2026-08-14 [codex] Validate Common Crawl WARC listing

Context:

- The official `CC-MAIN-2026-25` `warc.paths.gz` listing was supplied for the
  first local `0_raw_cc` pilot.

Commands:

```bash
cd web-curation-lab
gzip -t outputs/data/0_raw_cc/source/warc.paths.gz
gzip -dc outputs/data/0_raw_cc/source/warc.paths.gz | wc -l
gzip -dc outputs/data/0_raw_cc/source/warc.paths.gz | sed -n '1,5p'
shasum -a 256 outputs/data/0_raw_cc/source/warc.paths.gz
git check-ignore -v outputs/data/0_raw_cc/source/warc.paths.gz
```

Artifacts:

- Local ignored path listing:
  `outputs/data/0_raw_cc/source/warc.paths.gz`
- SHA-256:
  `20c2c05d3fa05a259b2791f379ea586a30034423f40fb00e7a18c170675d1753`

Result:

- Gzip integrity validation passed.
- The listing contains exactly 100,000 WARC paths and is ignored by Git under
  `/outputs/`.
- No WARC payload file has been downloaded.

Next:

- Record a deterministic pilot selection rule and download exactly one listed
  WARC file with resumable HTTPS transfer.

## 2026-08-14 [codex] Validate first pilot WARC

Context:

- The first entry in the official `CC-MAIN-2026-25` WARC listing was selected
  as the deterministic correctness input. This selection is not claimed to be
  statistically representative.

Commands:

```bash
cd web-curation-lab
stat -f '%z bytes' outputs/data/0_raw_cc/source/CC-MAIN-20260605214811-20260606004811-00000.warc.gz
gzip -t outputs/data/0_raw_cc/source/CC-MAIN-20260605214811-20260606004811-00000.warc.gz
shasum -a 256 outputs/data/0_raw_cc/source/CC-MAIN-20260605214811-20260606004811-00000.warc.gz
git check-ignore -v outputs/data/0_raw_cc/source/CC-MAIN-20260605214811-20260606004811-00000.warc.gz
```

Artifacts:

- Local ignored WARC:
  `outputs/data/0_raw_cc/source/CC-MAIN-20260605214811-20260606004811-00000.warc.gz`
- Size: `940246254` bytes
- SHA-256:
  `d17e1fd199541aba28c5ba3b7834319aa53e1dfaed2679eb83bbc86a19a663e0`

Result:

- The local size exactly matched the remote `Content-Length`.
- Gzip integrity validation passed.
- The WARC is ignored by Git under `/outputs/`.

Next:

- Read a bounded number of response records through vendored DataTrove and
  inspect the emitted text and metadata before defining `0_raw_cc` behavior.

## 2026-08-14 [codex] Implement bounded raw-WARC probe

Context:

- DataTrove 0.9.0's standard `WarcReader` accepts only HTML/XHTML response
  payloads (plus plain-text WET conversions), so it cannot define `0_raw_cc`
  without first measuring the records it silently excludes.
- The probe reads below the standard reader with `warcio`, preserves raw record
  order, emits short local previews, and mirrors the standard reader's keep/drop
  policy for comparison.

Commands:

```bash
cd web-curation-lab
uv run --extra data ruff check src/web_curation_lab/pipeline/cli.py src/web_curation_lab/pipeline/stages tests/pipeline/test_raw_cc_probe.py
uv run --extra data pytest tests/pipeline/test_raw_cc_probe.py -q
uv run --extra data web-curation-data inspect --config configs/data/0_raw_cc.toml --limit 100
uv run --extra data pytest -q
uv lock --check
git diff --check
```

Artifacts:

- Config: `configs/data/0_raw_cc.toml`
- CLI: `src/web_curation_lab/pipeline/cli.py`
- Stage implementation:
  `src/web_curation_lab/pipeline/stages/raw_cc/probe.py`
- Tests: `tests/pipeline/test_raw_cc_probe.py`
- Ignored records: `outputs/data/0_raw_cc/pilot/records.jsonl`
- Ignored summary: `outputs/data/0_raw_cc/pilot/summary.json`

Result:

- The first focused test run failed with
  `ImportError: failed to find libmagic. Check your installation`. DataTrove's
  `io` extra installs the Python binding but not the native macOS library. The
  probe now treats byte-level MIME sniffing as an optional reported capability;
  WARC and HTTP MIME metadata remain available without a system installation.
- The first real run exposed request targets being reported as status codes.
  Status parsing is now response-only, with a synthetic request regression test.
- The corrected 100-record prefix contains one `warcinfo`, 33 request, 33
  response, and 33 metadata records. All 33 responses are HTTP 200 HTML/XHTML
  and match DataTrove's standard keep policy. The probe inspected 1,202,783
  payload bytes.
- Generated artifacts contain exactly 100 records and are ignored by Git.
- Validation passed: focused tests 3 passed; full local tests 15 passed and 12
  expected macOS/Triton skips; Ruff, uv lock, and diff checks passed.

Next:

- Inspect a larger response sample before freezing textual MIME and decoding
  behavior; 33 homogeneous HTML responses are insufficient to define the
  complete `0_raw_cc` contract.

## 2026-08-14 [codex] Enable native MIME detection

Context:

- The user installed Homebrew `libmagic`, satisfying the native dependency
  behind uv-managed `python-magic`.

Commands:

```bash
cd web-curation-lab
uv run --extra data python -c 'import magic; print(magic.from_buffer(b"<html></html>", mime=True))'
uv run --extra data web-curation-data inspect --config configs/data/0_raw_cc.toml --limit 100
```

Artifacts:

- Regenerated ignored records: `outputs/data/0_raw_cc/pilot/records.jsonl`
- Regenerated ignored summary: `outputs/data/0_raw_cc/pilot/summary.json`

Result:

- The import smoke returned `text/html`; the regenerated summary records
  `libmagic_available: true`.
- Byte detection classified the 33 responses as 32 `text/html` and one
  `text/javascript`. The discrepant response declares HTML in both WARC and
  HTTP metadata and decodes strictly as UTF-8, so this remains a MIME-sniffing
  observation rather than a filter decision.
- The 34 `text/plain` detections are 33 WARC metadata payloads plus the single
  WARC-info payload; they are not HTTP response documents.

Next:

- Run a larger response-bounded inspection to measure MIME disagreements and
  non-HTML textual response types before defining the stage policy.

## 2026-08-14 [codex] Full one-WARC census

Context:

- DataTrove provides execution, sharding, batched token counting, and merged
  statistics, but its standard HTML-only WARC reader does not match
  `0_raw_cc`. Project code now owns the minimally decoded textual-response
  reader while reusing the rest of DataTrove.
- The textual MIME and decoding rules in this census are provisional measurement
  policy, not a frozen training-data contract.

Commands:

```bash
cd web-curation-lab
uv run --extra data pytest tests/pipeline/test_raw_cc_probe.py tests/pipeline/test_raw_cc_reader.py -q
uv run --extra data web-curation-data census --config configs/data/0_raw_cc.toml
uv run --extra data ruff check src/web_curation_lab/pipeline/cli.py src/web_curation_lab/pipeline/stages/raw_cc tests/pipeline
uv run --extra data pytest -q
uv lock --check
git diff --check
```

Artifacts:

- Reader: `src/web_curation_lab/pipeline/stages/raw_cc/reader.py`
- Census orchestration: `src/web_curation_lab/pipeline/stages/raw_cc/census.py`
- Config: `configs/data/0_raw_cc.toml`
- Ignored canonical summary: `outputs/data/0_raw_cc/census/summary.json`
- Ignored DataTrove stats and logs:
  `outputs/data/0_raw_cc/census/datatrove/`

Result:

- The first census launch failed before reading data with
  `TypeError: cannot pickle 'module' object`: the reader stored the imported
  `magic` module, while DataTrove deep-copies local pipelines. The reader now
  stores only serializable capability metadata and imports `magic` within each
  task; a deepcopy regression assertion passes.
- The corrected full census processed 63,412 WARC records: one WARC-info plus
  21,137 each of request, response, and metadata. All responses were HTTP 200.
- Provisional textual routing produced 20,919 candidates and 20,770 decoded
  documents; 218 responses were non-text MIME, 63 failed strict decoding, and
  86 decoded empty. Decoded-document retention was 98.2637% of responses.
- Response payloads totaled 3,963,731,964 bytes; retained source payloads
  totaled 3,704,572,647 bytes and decoded UTF-8 representations totaled
  3,704,621,759 bytes.
- The pinned Mistral tokenizer counted 1,626,767,305 content tokens. Adding one
  EOS boundary per document yields 1,626,788,075 provisional training tokens.
  At this exact file's yield, 61 files remain below 100B and 62 yield
  100,860,860,650 tokens. This is an extrapolation, not a frozen source count.
- End-to-end census wall time was 166.78 seconds on the local Mac. The
  tokenizer block accounted for 129.89 seconds; no text or token shards were
  materialized.
- Validation passed: focused reader/probe tests 4 passed; full local tests 16
  passed with 12 expected macOS/Triton skips; Ruff, uv lock, and diff checks
  passed.

Next:

- Audit the MIME disagreements and strict decode failures, then census a small
  deterministic spread of WARC files to measure token-yield variance before
  freezing the `0_raw_cc` inclusion contract or 100B inventory.

## 2026-08-14 [codex] Audit raw-WARC MIME and decoding decisions

Context:

- Added a deterministic full-WARC audit that uses the exact classifier shared
  with `RawWarcReader`; the audit does not maintain an independent copy of MIME
  or decoding behavior.
- The audit records complete category counts and bounded bottom-k SHA-256
  samples. Payload previews are capped and safely encoded. No training text is
  materialized and no reader policy changed.

Commands:

```bash
cd /path/to/web-curation-lab
uv run --extra data pytest tests/pipeline/test_raw_cc_audit.py tests/pipeline/test_raw_cc_reader.py tests/pipeline/test_raw_cc_probe.py -q
uv run --extra data ruff check src/web_curation_lab/pipeline/cli.py src/web_curation_lab/pipeline/stages/raw_cc tests/pipeline
uv run --extra data web-curation-data audit --config configs/data/0_raw_cc.toml
uv run --extra data pytest -q
uv lock --check
git diff --check
```

Artifacts:

- Audit implementation:
  `src/web_curation_lab/pipeline/stages/raw_cc/audit.py`
- Shared reader decision path:
  `src/web_curation_lab/pipeline/stages/raw_cc/reader.py`
- Tracked config: `configs/data/0_raw_cc.toml`
- Ignored audit summary: `outputs/data/0_raw_cc/audit/summary.json`
- Ignored deterministic samples: `outputs/data/0_raw_cc/audit/records.jsonl`

Result:

- The final audit re-read all 21,137 responses in 22.34 seconds and reproduced the
  census outcomes exactly: 20,770 accepted, 218 non-text MIME drops, 63 strict
  decode failures, and 86 empty-text drops.
- All 86 empty documents have zero-byte payloads. Of the 218 non-text records,
  202 are PDFs; the remainder are primarily other binary document/archive
  formats, with three BibTeX and one EndNote-classified record among them.
- There are 1,981 three-source MIME disagreements. Of these, 1,950 are accepted,
  13 overlap strict decoding failures, and 18 are non-text drops. Most are
  harmless taxonomy differences such as WARC `application/xhtml+xml`, HTTP
  `text/html`, and libmagic `text/html`.
- Following the reader's safe encoding priority (declared charset, UTF-8, then
  detected encoding), replacement-decoding diagnostics show 34 of 63 failures
  require at most 0.01% replacement characters, another 16 at most 0.1%, five
  at most 1%, and eight exceed 1%. An additional optimistic lower bound is
  recorded but is not a valid selection rule because minimizing replacement
  characters can choose a confidently wrong legacy encoding.
- The audit writes exactly 50 deterministic samples per category (200 rows),
  with 256-byte encoded payload prefixes and 500-character decoded previews.
- Validation passed: six focused tests, Ruff, the full local suite with 18
  passes and 12 expected macOS/Triton skips, uv lock verification, and diff
  whitespace checks.

Next:

- Decide whether `0_raw_cc` keeps strict decoding or uses a documented
  replacement fallback. The audit supports retaining the current WARC-first
  MIME precedence and excluding binary PDFs/documents.
- Then select and census a deterministic spread of roughly ten WARC files to
  measure token-yield variance before freezing the 100B source inventory.

## 2026-08-14 [codex] Freeze best-effort decoding policy v1

Context:

- The raw baseline should preserve noisy textual payloads rather than introduce
  an arbitrary replacement-character quality threshold. Policy revision
  `0_raw_cc-v1` therefore tries the declared charset, UTF-8, and detected
  encoding strictly; only if all strict attempts fail does it decode with
  replacement using the first valid encoding in that same priority order.
- PDFs and other binary MIME types remain excluded, and zero-byte textual
  responses remain empty drops. No HTML extraction, normalization, language
  filtering, deduplication, or quality filtering occurs.

Commands:

```bash
cd /path/to/web-curation-lab
uv run --extra data pytest tests/pipeline/test_raw_cc_audit.py tests/pipeline/test_raw_cc_reader.py tests/pipeline/test_raw_cc_probe.py -q
uv run --extra data ruff check src/web_curation_lab/pipeline/cli.py src/web_curation_lab/pipeline/stages/raw_cc tests/pipeline
uv run --extra data web-curation-data audit --config configs/data/0_raw_cc.toml
uv run --extra data web-curation-data census --config configs/data/0_raw_cc.toml
uv run --extra data pytest -q
uv lock --check
git diff --check
```

Artifacts:

- Policy config: `configs/data/0_raw_cc.toml`
- Reader and document metadata:
  `src/web_curation_lab/pipeline/stages/raw_cc/reader.py`
- Canonical ignored census:
  `outputs/data/0_raw_cc/census/summary.json`
- Canonical ignored audit: `outputs/data/0_raw_cc/audit/summary.json`
- Ignored deterministic audit samples:
  `outputs/data/0_raw_cc/audit/records.jsonl`

Result:

- Every output now records `policy_revision = "0_raw_cc-v1"`; retained
  documents record whether replacement was used and the exact replacement
  character count.
- All 63 former strict-decode failures are retained. They contain 28,492 total
  replacement characters. There are no remaining decode failures; the 218
  binary-MIME and 86 zero-byte decisions are unchanged.
- The revised census contains 20,833 documents, 1,630,359,019 Mistral content
  tokens, and 1,630,379,852 training tokens after one EOS boundary per document.
  This is 3,591,777 training tokens (+0.220789%) above the strict-decoding
  census.
- At this single file's revised yield, 61 files extrapolate to 99,453,170,972
  tokens and 62 to 101,083,550,824 tokens. This remains an extrapolation, not a
  source inventory.
- The final provenance-tagged census completed in 176.41 seconds. A preceding
  run produced identical document, replacement, byte, and token totals, showing
  deterministic accounting; runtime is not treated as a performance benchmark.
- Validation passed: six focused tests, Ruff, the full local suite with 18
  passes and 12 expected macOS/Triton skips, uv lock verification, and diff
  whitespace checks.

Next:

- Select a deterministic spread of roughly ten paths from the official
  100,000-file WARC listing, write the selection manifest and resumable download
  path, then census the sample under `0_raw_cc-v1` to measure yield variance.

## 2026-08-14 [codex] Complete deterministic ten-WARC yield pilot

Context:

- Added one resumable pilot command that selects paths, downloads and validates
  WARC gzip files, reuses matching census artifacts by input hash and policy
  revision, runs missing censuses, and derives a combined yield summary.
- Selection uses rounded evenly spaced zero-based indices including both
  endpoints. For the 100,000-entry `CC-MAIN-2026-25` listing, the one-based
  lines are 1, 11,112, 22,223, 33,334, 44,445, 55,556, 66,667, 77,778,
  88,889, and 100,000. This spans crawl segments dated June 5 through June 18.

Commands:

```bash
cd web-curation-lab
uv run --extra data pytest tests/pipeline/test_raw_cc_pilot.py tests/pipeline/test_raw_cc_audit.py tests/pipeline/test_raw_cc_reader.py tests/pipeline/test_raw_cc_probe.py -q
uv run --extra data ruff check src/web_curation_lab/pipeline/cli.py src/web_curation_lab/pipeline/stages/raw_cc tests/pipeline
uv run --extra data web-curation-data pilot --config configs/data/0_raw_cc.toml
uv run --extra data pytest -q
uv lock --check
git diff --check
```

Artifacts:

- Pilot implementation:
  `src/web_curation_lab/pipeline/stages/raw_cc/pilot.py`
- Tracked pilot config: `configs/data/0_raw_cc.toml`
- Ignored selection manifest:
  `outputs/data/0_raw_cc/pilot_10/selection_manifest.json`
- Ignored download manifest:
  `outputs/data/0_raw_cc/pilot_10/downloads_manifest.json`
- Ignored census resume manifest:
  `outputs/data/0_raw_cc/pilot_10/census_manifest.json`
- Ignored combined summary: `outputs/data/0_raw_cc/pilot_10/summary.json`
- Ignored per-WARC summaries and DataTrove logs:
  `outputs/data/0_raw_cc/pilot_10/census/`

Result:

- All ten selected WARCs are present, gzip-valid, and uniquely hashed. Nine were
  downloaded and the existing first WARC was reused. Compressed input totals
  9,163,298,804 bytes; local filesystem usage is 8.6 GiB.
- All ten censuses completed under `0_raw_cc-v1`; nine new results were written
  and the matching first-file result was reused by SHA-256 and policy revision.
- The sample contains 207,168 decoded documents and 15,983,361,896 training
  tokens including EOS boundaries.
- Per-WARC training-token yield has mean 1,598,336,189.6, median
  1,599,082,215.5, minimum 1,548,736,287, maximum 1,630,379,852, population
  standard deviation 25,186,105.79, and coefficient of variation 1.5758%.
- At the measured sample mean, 62 WARCs estimate to 99,096,843,755.2 tokens and
  63 estimate to 100,695,179,944.8. The pilot therefore changes the planning
  estimate from 62 files based on one WARC to 63 files based on ten WARCs.
- This is still an estimate. The full source inventory must include a safety
  buffer and be finalized from measured cumulative document/token totals; the
  training boundary will later be capped at a complete packed-sequence/global-
  batch boundary.
- Validation passed: Ruff, four pilot-specific tests, the full local suite with
  22 passes and 12 expected macOS/Triton skips, uv lock verification, and diff
  whitespace checks.

Next:

- Define the deterministic full-inventory selection and safety buffer, then
  choose where to run the roughly 63-WARC download/census and later token
  materialization. Do not provision it until storage format and batch-aligned
  token-capping behavior are specified.
