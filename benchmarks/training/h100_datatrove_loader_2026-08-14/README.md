# Two-H100 DataTrove loader validation

This directory preserves the derived measurements for the project-owned
DataTrove-to-TorchTitan loader at commit `1f2732d8`. Runs used two 80GB H100
GPUs, pure replicated data parallelism, local batch 80, global batch 160,
sequence length 2,048, BF16 parameters, FP32 gradient reduction,
`torch.compile`, and the mixed Muon/AdamW optimizer.

The metric window is steps 10 through 100 inclusive (91 measurements), so it
excludes compilation and initial warmup. TensorBoard scalar sums were reduced
to the sanitized statistics in `summary.json`; raw node-local artifacts remain
under `outputs/datatrove_loader_smoke/` and are intentionally ignored because
structured logs contain machine identifiers.

Three inputs isolate loader cost from token-content effects:

- `hf_control_100steps` uses the existing Hugging Face text loader.
- `natural_100steps` uses the DataTrove loader over tokens drawn from the fixed
  Paloma health corpus and repeated only to fill the bounded fixture.
- `uniform_100steps` uses the same DataTrove loader over an artificial uniform
  ID distribution. It is retained to show that unrealistic token distributions
  can change GPU-side embedding and loss performance even when data wait is
  negligible.

The realistic DataTrove run measured 385,201 global tokens/s and 45.645% MFU,
1.19% below the HF control's 389,851 tokens/s. Mean loader wait was 0.000113
seconds per step (0.0265%), slightly below the HF control. The loader is not an
end-to-end throughput bottleneck at this scale.

A separate checkpoint integration ran six steps with full checkpoints at steps
3 and 6, then restarted with a 12-step target. TorchTitan loaded `step-6` and
started at step 7. Exact next-batch behavior is additionally covered by the
single- and multi-worker unit tests.

Command pattern:

```bash
cd /path/to/web-curation-lab
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
```
