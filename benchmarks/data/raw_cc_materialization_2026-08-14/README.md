# One-WARC raw-CC materialization pilot

This directory preserves sanitized evidence from the first real
`0_raw_cc-v1` WARC materialization at commit `81fabf96`. The source was the
first entry of the deterministic `CC-MAIN-2026-25` pilot inventory.

The run verified the source inventory hash, reused the frozen raw-payload
reader, tokenized with the pinned Mistral tokenizer, appended EOS to every
document, shuffled deterministically inside the WARC, and streamed the result
into 2,049-token-aligned DataTrove shards. The final incomplete physical sample
contained 454 tokens and was discarded explicitly.

The pilot caught and corrected a pre-existing census accounting mistake. The
Mistral tokenizer's default postprocessor adds BOS, so the old census's
"content tokens" included one BOS per document and then added another EOS per
document. Actual training serialization disables default special tokens and
adds exactly one EOS. The corrected training total is therefore 1,630,359,019
tokens, exactly matching the materialized per-WARC token stream.

The initial assembly also inferred boundaries by scanning token ID 2. Literal
`</s>` text can encode to that ID, so this overcounted true document ends. The
accepted schema-v2 assembly carries DataTrove's exact document index instead.
It retains 20,832 complete document ends; the final document end lies in the
discarded 454-token tail.

The accepted two-GPU view contains 795,680 samples: 4,973 complete global
batches of 160 and 1,629,552,640 predicted tokens. Five additional complete
samples remain outside that batch-aligned view.

See `summary.json` for exact hashes and counts. Node-local raw manifests, logs,
token shards, and the training view remain under `outputs/data/0_raw_cc/`.
