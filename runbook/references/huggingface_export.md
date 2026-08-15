# Hugging Face Export Reference

TorchTitan can save the final Web Curation Lab Qwen3 model directly as Hugging
Face-compatible safetensors. The project model uses TorchTitan's
`Qwen3StateDictAdapter`, so a custom weight converter is not required.

## Final-Save Flags

Add these checkpoint settings to the production run:

```bash
--checkpoint.enable \
--checkpoint.folder ./outputs/<run-id>/checkpoints \
--checkpoint.last_save_model_only \
--checkpoint.last_save_in_hf \
--checkpoint.export_dtype bfloat16
```

`last_save_in_hf` requires a model-only last save. Intermediate resumable
checkpoints can remain in TorchTitan's distributed checkpoint format. The HF
save target must be a local filesystem path; upload or copy it afterward.

## Required Packaging

TorchTitan exports weights but does not create `config.json`. A complete model
directory also needs:

- a Qwen3 `config.json` matching the project architecture;
- the tokenizer files from `assets/hf/Mistral-7B-v0.1`;
- generation metadata only if the evaluation/inference consumer requires it.

For `curation_qwen3_150m_wide`, the important config values are:

```text
model_type: qwen3
vocab_size: 32000
hidden_size: 1024
intermediate_size: 3072
num_hidden_layers: 10
num_attention_heads: 8
num_key_value_heads: 2
head_dim: 128
max_position_embeddings: 4096
rope_theta: 1000000
rms_norm_eps: 1e-6
tie_word_embeddings: true
```

Read BOS/EOS/pad IDs from the pinned tokenizer assets rather than guessing
them.

The implementation command is:

```bash
uv run web-curation-package-hf \
  --checkpoint-dir outputs/<run>/checkpoints/step-<final> \
  --output-dir outputs/<run>/hf \
  --architecture qwen3_150m_wide \
  --run-id <run-id> --stage <stage> \
  --step <final-step> --tokens-seen <tokens>
```

## Validation Checklist

1. Save a small final checkpoint with native HF export enabled.
2. Generate and copy `config.json` plus tokenizer assets into the export.
3. Load the directory through the intended Transformers model class.
4. Compare representative state-dict keys and shapes.
5. Run identical token IDs through TorchTitan and the HF reload in evaluation
   mode and compare logits within a tolerance appropriate for the export dtype.
6. Generate a short greedy sample through the evaluation/inference path.
7. Record commands, tolerances, and artifacts in
   `runbook/streams/training_and_architecture.md`.

Steps 3–6 are implemented by the CUDA-only command below. It requires exact
argmax agreement and KL divergence no greater than `1e-6`, then separately
loads BF16 and performs greedy generation.

```bash
uv sync --extra eval-preflight
uv run --extra eval-preflight web-curation-check-hf-parity \
  --model outputs/<run>/hf \
  --output outputs/<run>/hf_parity.json
```

## Sources In This Repository

- `third_party/torchtitan/docs/checkpoint.md`
- `third_party/torchtitan/docs/evaluation.md`
- `third_party/torchtitan/torchtitan/components/checkpoint.py`
- `third_party/torchtitan/torchtitan/models/qwen3/state_dict_adapter.py`
- `src/web_curation_lab/training/models/qwen3.py`
