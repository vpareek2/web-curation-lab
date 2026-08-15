# Cloud Runs

Purpose: track cloud setup, benchmark and training queues, environment locks,
artifact retrieval, failures, and measured GPU results without recording
private connection details.

## 2026-08-14 [codex] Complete one-WARC raw-CC materialization pilot

Context:

- Copied the frozen ten-WARC inventory manifest to the active node, downloaded
  only its first WARC directly from Common Crawl, and verified its byte count
  and SHA-256 before processing.
- Ran the production materializer with `--limit 1`, corrected a BOS/EOS census
  accounting error and literal-EOS boundary inference exposed by the result,
  then regenerated only the final assembly from the valid resumable per-WARC
  artifact under schema v2.

Commands:

```bash
cd /path/to/web-curation-lab
curl --fail --location --retry 5 --continue-at - \
  <first-frozen-inventory-url> \
  --output outputs/data/0_raw_cc/source/<warc-filename>
uv run --frozen --extra data web-curation-data materialize \
  --config configs/data/0_raw_cc.toml \
  --limit 1
uv run --frozen --extra data web-curation-data create-training-view \
  --dataset-dir \
    outputs/data/0_raw_cc/training/shards/materialized-e291c9f520c8 \
  --output outputs/data/0_raw_cc/training/views/2xh100_one_warc.json \
  --sequence-length 2048 \
  --global-batch-size 160 \
  --training-steps 4973 \
  --seed 42
```

Artifacts:

- Tracked sanitized evidence:
  `benchmarks/data/raw_cc_materialization_2026-08-14/`
- Node-local source and inventory:
  `outputs/data/0_raw_cc/source/` and
  `outputs/data/0_raw_cc/pilot_10/downloads_manifest.json`
- Node-local resumable tokens and accepted schema-v2 shards:
  `outputs/data/0_raw_cc/training/tokenized/` and
  `outputs/data/0_raw_cc/training/shards/materialized-e291c9f520c8/`
- Node-local materialization and view manifests:
  `outputs/data/0_raw_cc/training/one_warc_materialize.json` and
  `outputs/data/0_raw_cc/training/views/2xh100_one_warc.json`

Result:

- The 940,246,254-byte source matched inventory SHA-256
  `d17e1fd199541aba28c5ba3b7834319aa53e1dfaed2679eb83bbc86a19a663e0`.
- DataTrove materialized 20,833 documents and 1,630,359,019 EOS-terminated
  tokens. Tokenization reached its shuffle phase after 313.377 seconds.
- The assembler retained 1,630,358,565 tokens in 795,685 complete physical
  samples across two shards and explicitly discarded a 454-token final tail.
- The accepted exact index contains 20,832 retained document ends. One final
  document end fell in the discarded incomplete-sample tail.
- The initial result revealed that the old census's content count included one
  default BOS per document and then added EOS again. Correct content without
  special tokens is 1,630,338,186; adding one EOS per document gives
  1,630,359,019, exactly the materialized source total.
- The first assembly also counted literal token ID 2 occurrences as document
  boundaries. Schema v2 now validates EOS at every DataTrove document end and
  carries the exact input indexes through assembly.
- A global-batch-160 view loaded with verified hashes and shapes `[2048]` for
  inputs and labels. It contains 4,973 steps, 795,680 samples, and
  1,629,552,640 predicted tokens; five complete samples remain unused.

Next:

- Update the ten-WARC census summaries under the corrected no-special-token
  contract or derive the exact one-BOS-per-document correction with explicit
  provenance, then decide whether to materialize the remaining nine pilot
  WARCs before freezing the larger source inventory.

## 2026-08-14 [codex] Validate stateful DataTrove loader on two H100s

Context:

- Validated the project-owned pretokenized loader under the selected Qwen3-wide
  model, local batch 80, and mixed Muon/AdamW optimizer.
- Compared a realistic-token DataTrove fixture with a same-node Hugging Face
  control. A uniform-ID fixture was also measured to distinguish input-content
  effects from loader wait.

Commands:

```bash
cd /path/to/web-curation-lab
uv sync --frozen --group dev --extra data
uv run --frozen --extra data pytest tests/training/test_datatrove_loader.py -q
uv run --frozen --extra data torchrun --standalone --nproc-per-node=2 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_scored \
  --dataloader.dataset-path \
    ./outputs/data/0_raw_cc/training/views/2xh100_natural_100_steps.json \
  --training.local-batch-size 80 \
  --training.global-batch-size 160 \
  --training.steps 100 \
  --parallelism.data-parallel-replicate-degree 2 \
  --checkpoint.no-enable \
  --validator.no-enable \
  --metrics.log-freq 1 \
  --dump-folder ./outputs/datatrove_loader_smoke/natural_100steps

# Checkpoint phase one; repeat with --training.steps 12 to resume step 6.
uv run --frozen --extra data torchrun --standalone --nproc-per-node=2 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_scored \
  --dataloader.dataset-path \
    ./outputs/data/0_raw_cc/training/views/2xh100_natural_100_steps.json \
  --training.local-batch-size 80 \
  --training.global-batch-size 160 \
  --training.steps 6 \
  --parallelism.data-parallel-replicate-degree 2 \
  --lr-scheduler.warmup-steps 1 \
  --checkpoint.interval 3 \
  --checkpoint.no-last-save-model-only \
  --checkpoint.no-last-save-in-hf \
  --validator.no-enable \
  --metrics.log-freq 1 \
  --dump-folder ./outputs/datatrove_loader_smoke/checkpoint_resume
```

- The first checkpoint attempt omitted the two-GPU topology overrides and
  failed with `Invalid parallel dims: dp_replicate(8) ... != WORLD_SIZE(2)`.
  The recorded command adds local/global batch 80/160 and DP replicate degree
  two; the production config itself correctly remains an eight-GPU recipe.

Artifacts:

- Tracked sanitized measurements:
  `benchmarks/training/h100_datatrove_loader_2026-08-14/`
- Node-local fixtures and frozen views:
  `outputs/data/0_raw_cc/training/smoke_shards_*` and
  `outputs/data/0_raw_cc/training/views/2xh100_*`
- Node-local TensorBoard, structured logs, and checkpoints:
  `outputs/datatrove_loader_smoke/`

Result:

- The Linux loader suite passed 15 tests, including exact single- and
  multi-worker resume, rank partitioning, corruption failures, and batched
  reads across shard boundaries.
- Across steps 10-100, the realistic DataTrove run averaged 385,201 global
  tokens/s, 45.645% MFU, and 0.000113 seconds of loader wait per step (0.0265%).
  The Hugging Face control averaged 389,851 tokens/s, 46.196% MFU, and 0.000129
  seconds of loader wait. DataTrove throughput was 1.19% lower, while measured
  data wait was slightly lower; the loader is not the bottleneck.
- The uniform-ID fixture averaged 374,817 tokens/s despite the same negligible
  loader wait. The artificial token distribution, rather than file I/O,
  explains most of its larger gap and is not representative training evidence.
- The first checkpoint run saved full state at step 6. The restarted process
  loaded `step-6` and began at step 7, exercising TorchTitan integration with
  the stateful DataTrove loader.
- Final local validation passed with 39 tests and 13 expected macOS/Triton
  skips; scoped Ruff, JSON parsing, and diff whitespace checks also passed.

Next:

- Implement the raw-WARC-to-DataTrove materializer and run it on one WARC on
  this node, preserving document boundaries, counts, shard hashes, and resume
  state. The production 100B shards and eight-GPU view are not created yet.

## 2026-08-12 [codex] Initial architecture comparison completed

Context:

- A 2xH100 cloud instance was used for the initial TorchTitan architecture
  comparison because the models use replicated data parallelism and fit on one
  GPU.
- The six approximately 150M configurations completed after environment and
  kernel compatibility fixes. The wide Qwen3 result was selected for the next
  phase.
- The exact output artifacts are not present in this local checkout.

Commands:

- No cloud command was run for this entry.
- Source: prior user/agent cloud session discussion and current git history.
- Connection commands and host details are intentionally omitted.

Artifacts:

- Relevant commits:
  - `c03a12bb` — architecture benchmark configs
  - `049c02ef` — prebuilt Hopper FlashAttention-3 support
  - `909b8639` — eager Qwen3.5 model path
  - `a838f8f0` — preload FA3 before Dynamo compilation
  - `d5cd95ba` — eager GPT-OSS FA3 model path
- Expected local training output root: `outputs/benchmarks/`
- Evidence requirements: `benchmarks/training/README.md`

Result:

- The comparison is complete for architecture selection purposes.
- Exact throughput, MFU, loss, environment versions, and peak-memory values are
  not restated because their source artifacts are absent here.
- Scaling from 2 to 8 GPUs is expected to be close to data-parallel scaling but
  must be measured; it is not guaranteed to be exactly proportional.

Next:

- Retrieve or rerun enough of the benchmark to preserve the raw logs and final
  summaries under `benchmarks/training/`.
- Before the 100B-token run, validate 8xH100 device topology, environment,
  tokenizer/data access, steady-state throughput, peak memory, checkpoint save
  and resume, final HF export, and artifact copy-back.
- Freeze CUDA, PyTorch, TorchTitan, FA3, and optional hybrid-kernel versions
  after that validation.

## 2026-08-13 [codex] Two-H200 checkpoint and evaluation preflight completed

Context:

- Ran the selected 153.38M-parameter wide Qwen3 on two H200 GPUs using pure
  replicated data parallelism and the fixed local Paloma health data.
- This was an integration run, not a throughput benchmark or useful model.

Result:

- Phase one trained through step 6 and wrote a complete resumable DCP
  checkpoint. Phase two explicitly resumed step 6 and finished at step 12,
  writing a native FP32 HF model-only export.
- Final training loss was 6.518853 and validation loss was 6.070210 after
  196,608 tokens. Steps 11-12 measured roughly 185k global tokens/s and 22%
  MFU; the tiny run and checkpoint interruptions make these unsuitable for a
  production ETA. Peak active/reserved memory was 5.04/6.12 GiB per GPU.
- HF parity, BF16 inference, distributed Paloma accounting, and the complete
  limited 83-task general smoke passed.
- Full Paloma/general runs were started only to inspect their scale, then
  intentionally stopped because a 12-step checkpoint does not justify their
  cost. All evaluation processes were terminated and both GPUs were released.
- A 5.3MB evidence bundle was copied to
  `benchmarks/training/h200_preflight_2026-08-13/`. Multi-gigabyte model and DCP
  files remain disposable node artifacts and need not be retained.

Next:

- The temporary node can be torn down.
- Before the 100B run, increase batch size on the target eight-GPU topology and
  measure a longer steady-state window after compilation.

## 2026-08-14 [codex] Two-H100 Muon preflight blocked on driver version

Context:

- Provisioned a two-H100 preparation node for mixed Muon/AdamW correctness,
  batch tuning, and raw-crawl processing preflight.
- Verified two full-power 80GB H100 SXM GPUs connected by bonded NVLink, 52
  vCPUs, 442 GiB RAM, and 5.4 TiB free local storage.
- Published and checked out project commit `f58e7b24`, then installed the
  frozen root uv environment.

Commands:

```bash
cd /path/to/web-curation-lab
uv sync --frozen --group dev
uv run --frozen python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Artifacts:

- Project commit: `f58e7b24`
- No training artifact was created because CUDA initialization failed before
  model construction.

Result:

- The lock installed PyTorch `2.13.0+cu130` as intended.
- CUDA initialization failed with: `The NVIDIA driver on your system is too
  old (found version 12080)`.
- The node has open server driver 570.148.08. Open server driver 580 is
  available from its configured package repositories; the official PyTorch
  CUDA 12.8 wheel index does not provide a PyTorch 2.13 build.

Next:

- Upgrade the node to the open 580 server driver and reboot, subject to user
  approval, then rerun CUDA initialization and the mixed-optimizer tests.
- Do not downgrade PyTorch merely to accommodate the image driver because that
  would change the frozen training stack.

## 2026-08-14 [codex] Two-H100 Muon checkpoint and HF parity preflight passed

Context:

- Upgraded the preparation node to an open 580 server driver and retained the
  frozen PyTorch `2.13.0+cu130` environment.
- Ran the selected 153.38M-parameter wide Qwen3 with pure two-GPU replicated
  data parallelism and the mixed Muon/AdamW optimizer.
- This twelve-step local-data run checks integration only. Its losses, MFU, and
  throughput are not production measurements.

Commands:

```bash
cd /path/to/web-curation-lab
uv run --frozen python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
uv run --frozen pytest tests/training/test_optimizer.py tests/training/test_reference_config.py -q
uv run --frozen torchrun --standalone --nproc-per-node=2 -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_preflight_checkpoint
uv run --frozen torchrun --standalone --nproc-per-node=2 -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_preflight_export
uv run --frozen web-curation-package-hf \
  --checkpoint-dir outputs/preflight/qwen3_150m_wide/checkpoints/step-12 \
  --output-dir outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --architecture qwen3_150m_wide \
  --run-id qwen3-wide-preflight \
  --stage preflight \
  --step 12 \
  --tokens-seen 196608
uv sync --frozen --extra eval-preflight
uv run --frozen --extra eval-preflight web-curation-check-hf-parity \
  --model outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --output outputs/preflight/qwen3_150m_wide/hf_parity.json
```

Artifacts:

- Node-local DCP checkpoints:
  `outputs/preflight/qwen3_150m_wide/checkpoints/step-{3,6,9}/`
- Node-local native FP32 HF checkpoint:
  `outputs/preflight/qwen3_150m_wide/checkpoints/step-12/`
- Node-local packaged model and manifest:
  `outputs/preflight/qwen3_150m_wide/hf-fp32/`
- Node-local parity report:
  `outputs/preflight/qwen3_150m_wide/hf_parity.json`
- Node-local structured logs:
  `outputs/preflight/qwen3_150m_wide/structured_logs/`

Result:

- CUDA initialized on both H100s and a BF16 matrix multiplication completed.
- TorchTitan routed 50 hidden two-dimensional parameter tensors to Muon and 42
  fallback tensors to fused AdamW. The model compiled and completed two-rank
  forward, backward, optimizer, validation, and checkpoint operations.
- Step 6 produced a full resumable DCP checkpoint. The second process loaded
  `step-6`, resumed at step 7, and completed at step 12.
- Peak reported memory was 5.73 GiB per GPU at local batch 4, leaving ample
  headroom for the required batch-size sweep.
- The final validation loss was 5.5088 after 196,608 tokens. This is smoke-test
  telemetry only because the fixed health data is reused for training here.
- HF packaging verified the pinned Mistral tokenizer revision, 32,000-token
  vocabulary, BOS ID 1, and EOS ID 2.
- Transformers parity passed: matching argmaxes, KL divergence
  `-2.201146998004333e-07` at a `1e-6` threshold, maximum absolute logit error
  `1.9073486328125e-06`, and four generated tokens.
- The selected Qwen3/Muon tests passed. Three unrelated GPT-OSS tests in the
  broader test file failed because the optional `kernels` package was absent;
  the selected dense Qwen3 path does not require it.

Next:

- Run a longer steady-state local-batch sweep on the two-H100 node, excluding
  compilation, validation, and checkpoint steps from throughput summaries.
- Repeat the chosen local batch briefly on the final eight-GPU topology before
  freezing global batch and generating the exact batch-aligned training view.

## 2026-08-14 [codex] Two-H100 Muon batch sweep selected local batch 80

Context:

- Swept local batch sizes for the selected Qwen3-wide model using the verified
  mixed Muon/AdamW optimizer on two H100s with replicated data parallelism.
- Used the fixed Paloma health JSONL smoke loader. The dataset repeatedly wraps,
  so the selected batch remains provisional until the pretokenized loader and
  final eight-GPU topology confirm it.

Commands:

```bash
cd /path/to/web-curation-lab
uv run --frozen torchrun --standalone --nproc-per-node=2 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_preflight_checkpoint \
  --training.local_batch_size <8|16|32|48|64|80|96> \
  --training.global_batch_size <2x-local-batch> \
  --training.steps 30 \
  --checkpoint.no-enable \
  --validator.no-enable \
  --metrics.log_freq 1 \
  --dump_folder ./outputs/batch_sweep/<run-name>
```

- Repeated the same command for local batches 64 and 80 with
  `--training.steps 100`.
- Derived metrics average duplicate rank lines for each step, then summarize
  steps 10 through the end.

Artifacts:

- Tracked sanitized logs and derived measurements:
  `benchmarks/training/h100_muon_batch_sweep_2026-08-14/`
- Node-local TensorBoard and structured logs:
  `outputs/batch_sweep/`

Result:

- Short-sweep means for local batches 32, 48, 64, 80, and 96 were respectively
  331k, 364k, 380k, 391k, and 397k global tokens/s.
- The 100-step confirmations measured 378,445 tokens/s and 44.84% MFU at
  local batch 64, versus 389,542 tokens/s and 46.16% MFU at local batch 80.
  Both summaries use 91 steps after discarding steps 1-9.
- Local batch 80 used 60.68 GiB per GPU and was 2.93% faster than local batch
  64. Local batch 96 used 72.45 of 79.18 GiB and was only about 1.4% faster
  than local batch 80 in the short sweep, so it was rejected as too close to
  the memory boundary.
- Set the current root training default to local batch 80. On eight GPUs this
  is global batch 640 and 1,310,720 predicted tokens per step. A 100B-token run
  requires 76,294 steps and processes 100,000,071,680 tokens.

Next:

- Implement and benchmark the pretokenized DataTrove loader on this node.
- Confirm local batch 80 on the final eight-GPU node before declaring the global
  batch frozen and creating the exact batch-aligned training view.
