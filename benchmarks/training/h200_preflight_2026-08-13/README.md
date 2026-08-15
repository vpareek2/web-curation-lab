# Two-H200 training and evaluation preflight

This directory contains the compact evidence retained from the 2026-08-13
Qwen3-wide preflight. Model weights and resumable checkpoints were deliberately
not copied from the temporary node.

## Training

- Hardware: 2x NVIDIA H200, replicated data parallelism.
- Model: `qwen3_150m_wide`, 153,378,304 parameters.
- Sequence length: 2,048; local batch: 4; global batch: 8.
- Phase one trained through step 6 and wrote a complete DCP checkpoint. Phase
  two resumed that checkpoint and trained through step 12 (196,608 tokens).
- Final loss: 6.518853; validation loss: 6.070210.
- Steps 11 and 12 measured 185,957 and 184,451 tokens/s globally, with 22.04%
  and 21.86% MFU. This run is too short for production throughput claims.
- Peak active/reserved memory in phase two: 5.04/6.12 GiB per GPU.

## Export and evaluation

- Native TorchTitan FP32 HF export packaged with the pinned Mistral tokenizer.
- Transformers/TorchTitan argmaxes matched; KL was effectively zero and the
  maximum absolute logit difference was 1.91e-6. BF16 loading and greedy
  generation succeeded.
- The 100-document Paloma check produced identical document, predicted-token,
  and UTF-8-byte totals on one and two GPUs. Loss differed by 2.21e-8 per token
  due to floating-point reduction order. Measured wall time was 5.02s versus
  3.49s.
- The limited OLMo Eval smoke completed all 83 tasks (415 instances), reported
  no errors, and took 118.86s including vLLM startup. Its aggregate score is
  not scientifically meaningful for a 12-step model.

The TensorBoard events and structured logs are the authoritative training
measurements. `general-smoke/general_raw.json`, `hf_parity.json`, and the two
statistics JSON files preserve evaluation results and provenance.
