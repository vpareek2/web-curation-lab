# Evaluation environment

This is a separate uv project so the Paloma dependency stack cannot alter the
TorchTitan training environment. The general suite executes from OLMo Eval's
own frozen uv environment, vendored at commit
`479f90ea137a2a044a33aa7c81128d400bd027c4`.

```bash
uv sync --project evaluation --group dev
uv run --project evaluation web-curation-eval prepare-assets \
  --config configs/evaluation/full.toml
uv run --project evaluation web-curation-eval run \
  --model outputs/<run>/hf \
  --config configs/evaluation/full.toml \
  --output outputs/evaluations/<run-id>
```

The general suite is `olmobase:easy:qa:rc`, the current OLMo 3 base-model QA
suite. It expands to 83 task configurations, so `easy` describes the capability
tier rather than a tiny runtime. Results are called OLMo Eval scores, not DCLM
`Core_v2`; the suites use different prompts, tasks, and aggregation.

Paloma is gated. Accept the AI2 license at
<https://huggingface.co/datasets/allenai/paloma> and authenticate with
Hugging Face before preparing assets.

Package TorchTitan's final native HF save with the exact project architecture,
pinned tokenizer, and a hashed evaluation manifest:

```bash
uv run web-curation-package-hf \
  --checkpoint-dir outputs/<run>/checkpoints/step-<final> \
  --output-dir outputs/<run>/hf \
  --architecture qwen3_150m_wide \
  --run-id <run-id> \
  --stage <stage> \
  --step <final-step> \
  --tokens-seen <tokens>
```

On a CUDA node, verify FP32 TorchTitan/Transformers parity plus BF16 loading and
greedy generation before the first scored run:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_preflight_checkpoint
uv run torchrun --standalone --nproc-per-node=2 -m torchtitan.train \
  --module web_curation_lab.training \
  --config curation_qwen3_150m_wide_preflight_export

uv run web-curation-package-hf \
  --checkpoint-dir outputs/preflight/qwen3_150m_wide/checkpoints/step-12 \
  --output-dir outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --architecture qwen3_150m_wide \
  --run-id qwen3-wide-preflight \
  --stage preflight \
  --step 12 \
  --tokens-seen 196608

uv sync --extra eval-preflight
uv run --extra eval-preflight web-curation-check-hf-parity \
  --model outputs/preflight/qwen3_150m_wide/hf-fp32 \
  --output outputs/preflight/qwen3_150m_wide/hf_parity.json
```

The parity command requires matching argmaxes and KL divergence at most
`1e-6`. The full evaluation directory contains `evaluation_manifest.json`,
`statistics.json`, `general_raw.json`, `summary.json`, `general_raw/`, and
`logs/`.

Run the full Paloma and general suites on the final model from every curation
stage and on the final 100B-token S0 model. For each later stage, train a
separate S0 baseline from the shared initialization using raw S0 data and that
stage's exact retained-token budget. Keep optimizer, global batch, sequence
length, seed, and WSD percentages identical. An ordinary checkpoint from the
100B S0 run is not a token-matched baseline because percentage-based WSD gives
it a different learning-rate trajectory.
