# Cloud Runs

Purpose: track cloud setup, benchmark and training queues, environment locks,
artifact retrieval, failures, and measured GPU results without recording
private connection details.

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
