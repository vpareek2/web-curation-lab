# Training And Architecture

Purpose: track architecture selection, training configuration, kernels,
throughput comparisons, checkpoint behavior, and final training-contract
decisions.

## 2026-08-12 [codex] Current architecture and native HF export

Context:

- Six approximately 150M-parameter recipes were implemented for an initial
  end-to-end TorchTitan comparison: Qwen3, dense GPT-OSS-style attention, and
  Qwen3.5 hybrid, each in 16-layer reference and 10-layer wide shapes.
- Based on the completed cloud comparison discussed with the user, the wide
  dense Qwen3 recipe is the current choice because of its throughput/quality
  tradeoff.
- Inspected TorchTitan checkpoint documentation, checkpoint implementation,
  and the Qwen3 state-dict adapter to verify export support.

Commands:

```bash
cd web-curation-lab
rg -n "last_save_in_hf|export_dtype|StateDictAdapter" \
  third_party/torchtitan/docs \
  third_party/torchtitan/torchtitan \
  src/web_curation_lab/training
```

No training or conversion command was run for this entry. The architecture
choice comes from the prior cloud run discussion; exact cloud result files are
not present in this checkout and therefore no benchmark numbers are recorded
here.

Artifacts:

- Selected config: `curation_qwen3_150m_wide`
- Architecture definition:
  `src/web_curation_lab/training/models/qwen3.py`
- TorchTitan configuration registry:
  `src/web_curation_lab/training/config_registry.py`
- TorchTitan checkpoint documentation:
  `third_party/torchtitan/docs/checkpoint.md`
- Qwen3 adapter:
  `third_party/torchtitan/torchtitan/models/qwen3/state_dict_adapter.py`
- Benchmark evidence contract: `benchmarks/training/README.md`

Result:

- Current model: 10 layers, dimension 1,024, 8 query heads, 2 KV heads,
  head dimension 128, SwiGLU hidden dimension 3,072, tied 32,000-token
  embeddings; calculated size 153.38M parameters.
- TorchTitan can write the final model directly as HF safetensors using
  `checkpoint.last_save_in_hf` with model-only saving.
- The project model already supplies TorchTitan's `Qwen3StateDictAdapter`; no
  custom state-dict exporter is required.
- TorchTitan does not create `config.json`. The final HF directory must add a
  project-generated Qwen3 config and the pinned tokenizer assets.
- Native HF export targets a local checkpoint folder, not a remote URI.
- The benchmark WSD defaults are 10% warmup and 15% cooldown. These fractions
  are current working values, not yet a frozen production decision.

Next:

- Add a production training config rather than repurposing the 500M-token
  benchmark config.
- Package a matching HF `config.json` and tokenizer files, then compare logits
  between the TorchTitan model and an HF reload from one small checkpoint.
- Recover/copy the initial cloud logs into `benchmarks/training/` before citing
  exact throughput, MFU, or final loss in the project report.
- On the production 8xH100 node, increase batch size until memory or throughput
  stops improving, then freeze the global batch and runtime estimate.

## 2026-08-12 [codex] Scored recipe and HF packaging implemented

Context:

- Added a separate 100B-token wide-Qwen3 scored recipe rather than changing the
  500M-token architecture benchmark.
- Added final HF packaging and a CUDA parity/generation preflight.
- Pinned the four tokenizer file hashes in addition to repo revision and token
  IDs.

Commands:

```bash
cd web-curation-lab
uv lock
uv run pytest -q
uv run ruff check \
  src/web_curation_lab/evaluation \
  src/web_curation_lab/tokenizer_assets.py \
  src/web_curation_lab/training/config_registry.py \
  tests/evaluation tests/training/test_tokenizer_assets.py
```

Artifacts:

- Scored config: `curation_qwen3_150m_wide_scored`
- Packager: `src/web_curation_lab/evaluation/hf_package.py`
- Parity preflight: `src/web_curation_lab/evaluation/hf_parity.py`
- Tokenizer contract: `src/web_curation_lab/tokenizer_assets.py`

Result:

- The scored recipe uses 100B target tokens, percentage WSD (10% warmup, 15%
  cooldown), resumable DCP checkpoints every 1%, and telemetry validation every
  2% on the prepared 2M-token Paloma health set.
- The final save is model-only HF safetensors in BF16; prior periodic DCP saves
  remain the recovery path.
- Root tests passed locally; GPU/Triton-dependent config and parity behavior are
  intentionally deferred to cloud preflight.

Next:

- On CUDA, export a short-run checkpoint and run
  `web-curation-check-hf-parity` before any scored training.
- Freeze production global batch size before calculating final steps and
  checkpoint/validation intervals; current values assume batch 128 at length
  2,048.
