Project Summary: Progressive Web-Data Curation for a Small Language Model
Project goal
This project will measure how model quality changes as a large, messy web corpus is progressively reduced through increasingly aggressive curation.
The central experiment is:
Can a 100–150M parameter language model trained on 5–15B highly curated tokens outperform the same model trained on 100B minimally processed web tokens?
Unlike a conventional data ablation where every model receives the same number of tokens, each model will train on all data remaining after its stage of the pipeline. This reflects the practical decision an LM developer faces: whether it is better to spend compute on a large mediocre corpus or discard most of it and train on a smaller, better one.
The main outputs will be quality-versus-tokens and quality-versus-training-cost curves, rather than merely a claim that “clean data is better.”

Starting corpus: where the 100B tokens come from
The recommended source is DCLM-Pool, rather than downloading a full Common Crawl month from scratch. DCLM-Pool contains extraction-only web text derived from Common Crawl using Resiliparse. The published pool contains 240T GPT-NeoX tokens across 200B documents and is 370TB after gzip compression. DCLM also publishes smaller raw competition pools that have undergone only text extraction; the current repository lists its smallest pool at 137B tokens. (arXiv)
The construction procedure would be:
	1	Select DCLM extraction-only shards deterministically.
	2	Apply only a permissive English-routing rule, not a quality filter.
	3	Apply the same evaluation-decontamination mask to every stage.
	4	Count tokens using the final project tokenizer, stopping when the corpus reaches 100B tokens.
This is important because “100B tokens” is tokenizer-dependent: DCLM reports GPT-NeoX tokens, while FineWeb reports GPT-2 tokens. The project’s 100B figure should always mean 100B tokens under the tokenizer actually used to train the model.
At this scale, the source text should be roughly 150–300GB compressed, depending on representation. A linear estimate from DCLM gives approximately 154GB, while FineWeb’s published 100B-token Parquet sample is 277.4GB. FineWeb should not be used as the raw baseline because it is already heavily processed, but its sample validates the storage estimate. (arXiv)
Starting from DCLM rather than WARC does mean that “raw” refers to raw extracted web text, not raw HTTP and HTML bytes. That is appropriate because the experiment is about curation after extraction. A recent Common Crawl month contains about 85TiB of compressed WARC and 364TiB uncompressed, making full WARC processing unnecessarily expensive for this PoC. A small WARC side experiment could still be included to demonstrate extraction quality. (Common Crawl)

Curation funnel
Every stage will be a nested subset of the previous one. Token counts below are planning ranges, not quotas; actual retention is one of the results.
Stage
Processing added
Expected scale
Training time target
S0: Extraction-only
Permissive English routing and decontamination only
100B
≤12 hr
S1: Hygiene
Strict language ID, malformed text, extreme lengths, boilerplate, repetition, symbol/markup and URL rules
60–80B
7–10 hr
S2: Deduplicated
Exact document/paragraph deduplication plus MinHash near-deduplication
30–60B
4–7 hr
S3: Heuristic quality
Selected Gopher, C4 and FineWeb-style content rules
15–35B
2–4 hr
S4: Learned quality
General-purpose quality classifier; retain the highest-quality slice
5–15B
1–2 hr
FineWeb provides a strong reference implementation for extraction, MinHash deduplication and heuristic filtering. Its pipeline uses WARC-based extraction, fuzzy deduplication based on document 5-grams and a combination of C4 and custom rules. The exact thresholds should first be audited on a one-billion-token pilot and then frozen before the full run. (arXiv)
For learned filtering, I would start with the DCLM general-quality classifier rather than make FineWeb-Edu the only final stage. FineWeb-Edu is useful as an optional branch, but its classifier explicitly targets educational material and can favor primary or grade-school-style content over specialized domains. (Hugging Face)
Each document should receive a stable ID. Rather than copying the full corpus after every stage, retain one canonical text store and write:
	•	filter scores and reasons;
	•	per-stage keep/drop manifests;
	•	duplicate-cluster IDs;
	•	document metadata such as crawl, domain and length.
Only materialize the tokenized training shards required for each run. This makes the pipeline cheaper, auditable and easy to modify.

Model-training design
A short Phase 0 will compare approximately 100M, 150M and possibly 250M configurations on tokens processed per dollar, not only raw model FLOPs. The architecture is frozen before running any curation experiment.
To process 100B tokens in 12 hours, the selected configuration must sustain:
[ \frac{100B}{12\text{ hours}} \approx 2.32\text{ million tokens/second} ]
across the eight-H100 node, or approximately 289,000 tokens/second per GPU. This should be treated as an acceptance test before committing to the full raw run.
The actual experiment then uses:
	•	identical model initialization, tokenizer and architecture;
	•	BF16, FlashAttention and DDP—FSDP is unnecessary at this scale;
	•	the same global batch size and optimizer;
	•	the same percentage-based WSD schedule (current defaults: 10% warmup and 15% cooldown);
	•	one pass over all tokens surviving each stage.
Because WSD phases are percentages of each run, equal-token S0 comparisons must be independent runs from the shared initialization. Ordinary checkpoints from the 100B S0 run have a different learning-rate history and are not valid token-matched baselines. Each matched run uses raw S0 data, the retained stage's exact token budget, and identical optimizer, batch, sequence length, seed, and WSD percentages. This produces two important comparisons:
	•	S4 at 8B versus S0 at 8B: quality at equal training compute.
	•	S4 at 8B versus S0 at 100B: quality versus realistic total cost.
At the target throughput, the five primary runs should require roughly 25–35 eight-H100 node-hours in total. A second seed should be reserved for S0 and S4 if the initial score difference is close enough that variance matters.

Benchmarking strategy
The primary metric should be held-out cross-entropy by domain, using Paloma. Paloma measures language-model fit across 546 English and code domains, making it more informative than one aggregate web-perplexity number. It should reveal whether a filter improves general web language while damaging code, forums, technical writing or other distributions. (arXiv)
The general benchmark metric uses the modern OLMo Eval `olmobase:easy:qa:rc` suite. It targets base-model evaluation and reports a reproducible average-of-averages across reading-comprehension formulations; the pinned version expands to 83 task configurations. Preserve every task score and raw prediction; the suite aggregate is useful for navigation, while individual tasks explain regressions. These results must not be called DCLM Core because the prompts, tasks, and aggregation differ.
The final report should show:
	1	Paloma bits/byte and OLMo Eval QA score versus training tokens.
	2	The same scores versus eight-H100 node-hours.
	3	Documents and tokens retained at every stage.
	4	Per-domain performance changes.
	5	Duplicate rate, document-length and quality-score distributions.
	6	Random samples of accepted and rejected documents for every filter.

Processing hardware and software
Recommended hardware
The data pipeline does not require the H100 node for most stages.
Resource
Recommended configuration
CPU processing
64–128 vCPU
Memory
256GB minimum; 512GB preferred
Local storage
4–8TB fast NVMe
Network
10Gbps or faster to object storage
Learned filtering
One GPU initially; scale to several only if inference becomes the bottleneck
LM training
8×H100
A mandatory 1B-token pilot should measure throughput, temporary-storage amplification and retention before provisioning the full run. Deduplication and shuffling are likely to be more I/O-sensitive than RAM-sensitive, which is why local NVMe is valuable.
Recommended software
Use DataTrove as the primary pipeline framework. It already supports large-scale text extraction, filtering, tokenization and deduplication; it can run locally, through Slurm or through Ray; and it includes a full FineWeb reproduction and Common Crawl WARC example. (GitHub)
The supporting stack would be:
	•	DCLM-Pool and its quality-classification artifacts;
	•	DataTrove for orchestration, statistics, filtering and MinHash;
	•	Resiliparse only for the optional WARC extraction experiment;
	•	fastText for language identification and inexpensive quality scoring;
	•	Hugging Face Tokenizers for final token accounting;
	•	Paloma plus the pinned OLMo Eval workbench for evaluation.
Dolma is a reasonable alternative if its Rust Bloom-filter deduplication is more convenient. NeMo Curator becomes attractive if the project expands enough to justify GPU-accelerated exact, fuzzy or semantic deduplication, but it is unnecessary for the initial 100B-token PoC. (GitHub)
The project should not implement WARC parsing, distributed MinHash, tokenizer orchestration or evaluation infrastructure from scratch. Custom code should be limited to deterministic corpus sampling, stage manifests, experimental filters and reporting.

Expected outcome
The ideal headline is:
A 100–150M model trained on approximately 10B curated web tokens outperforms the same model trained on 100B extraction-only tokens, using roughly one-tenth the training compute.
Even if that exact crossover does not occur, the project still produces a useful result: the break-even point where further filtering begins to remove more learning signal than noise, plus a concrete map of which capabilities are helped or harmed by each curation stage.
