# Two-H100 Muon batch sweep

This directory preserves the text logs and derived summary for the selected
153,378,304-parameter Qwen3-wide model on two 80GB H100 GPUs. The topology was
pure replicated data parallelism. Training used BF16 parameters, FP32 gradient
reduction, `torch.compile`, and the project mixed Muon/AdamW optimizer.

Each initial point ran for 30 steps. The two confirmation points ran for 100
steps. Derived statistics average the two rank log lines for each step and then
summarize steps 10 through the end, excluding compilation and early warmup.

The sweep used the fixed Paloma health JSONL through TorchTitan's text loader.
It therefore measures the current end-to-end smoke path, not the final
pretokenized training loader. The dataset repeatedly wraps during these runs.
The chosen batch must be confirmed with the pretokenized loader and on the
final eight-GPU topology.

The current throughput default is local batch 80. On eight GPUs this gives
global batch 640, 1,310,720 predicted tokens per optimizer step, and 76,294
steps for at least 100B tokens. Local batch 96 was rejected despite fitting:
it used 72.45 of 79.18 GiB per GPU and improved throughput by only about 1.4%
over local batch 80 in the short sweep.

Command pattern:

```bash
cd /path/to/web-curation-lab
uv run --frozen torchrun --standalone --nproc-per-node=2 \
  -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_preflight_checkpoint \
  --training.local_batch_size <local-batch> \
  --training.global_batch_size <2x-local-batch> \
  --training.steps <30-or-100> \
  --checkpoint.no-enable \
  --validator.no-enable \
  --metrics.log_freq 1 \
  --dump_folder ./outputs/batch_sweep/<run-name>
```

See `summary.json` for the derived measurements and `logs/` for the sanitized
raw text logs.
