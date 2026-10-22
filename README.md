# Data for "When Do Query Rewriting, Hybrid Search, and Reranking Hurt?"

Data accompanying the paper *When Do Query Rewriting, Hybrid Search, and Reranking Hurt? A Component Ablation of Retrieval for Indonesian Civil-Service Regulation RAG* (Moh. Affan, Gunawan, Lukman Zaman; Institut Sains dan Teknologi Terpadu Surabaya). Indonesian title: *Kapan Query Rewriting, Hybrid Search, dan Reranking Merugikan? Ablasi Komponen Retrieval pada RAG Regulasi Kepegawaian Indonesia*.

The package contains the test set, every retrieved context and answer for each retrieval arm, per-question scores, the relevance judgments for real user queries (§4.8), and the LLM-free reranker diagnosis (§4.6). The system code and experiment scripts are owned by BKPSDM Kabupaten Sumenep and are not included; they are available from the corresponding author on request, subject to permission from BKPSDM Kabupaten Sumenep.

The fine-tuned reranker and its training data are published separately on Hugging Face: [moh-affan/indobert-reranker-asn-sumenep](https://huggingface.co/moh-affan/indobert-reranker-asn-sumenep) and [moh-affan/reranker-asn-sumenep-dataset](https://huggingface.co/datasets/moh-affan/reranker-asn-sumenep-dataset). The test set is the same as in the companion chunking study ([legal-rag-chunking-ablation-data](https://github.com/bkpsdm-sumenep/legal-rag-chunking-ablation-data)).

## Contents

| Path | Content |
|:---|:---|
| `data/testset.jsonl` | 109 validated test items plus the 4 items excluded after the runs (`excluded_after_run`) |
| `data/arms.json` | Description, item set, and internal run ID of every arm |
| `data/outputs/<arm>.jsonl` | One line per question: search query, retrieved contexts with metadata, answer, scores, retrieval metrics |
| `data/scores.csv` | Scores only, one row per arm and question |
| `real_queries/queries.csv` | 80 sampled real user queries (codes only): length and scope decisions |
| `real_queries/judgments.csv` | Relevance judgments (0/1/2) per query-article pair and each arm's top-5 rank |
| `real_queries/pool_meta.json` | Sampling and pooling settings |
| `real_queries/results_table11_strict_scope.json` | Table 11 metrics (60 queries in scope under the strict scope decision) |
| `real_queries/results_original_scope.json` | Same metrics with the original scope decision (66 queries) |
| `diagnostics/reranker_diagnosis.json` | LLM-free reranker diagnosis on the 97 in-scope items (paper §4.6, Table 9) |
| `results/verification.json` | Means and paired differences recomputed from the files in this package |

## Arms

| Arm | Items | Configuration |
|:---|:---|:---|
| `Baseline` | all | Dense retrieval only, no query rewriting |
| `Treatment` | all | Query rewriting + hybrid search (BM25), RRF 1:1, k = 60 |
| `X0` | all | Repeat of the baseline (run-to-run variation) |
| `X1` | all | Rewriting only (dense) |
| `X2` | all | Hybrid BM25 only, RRF 1:1, k = 60 |
| `X3` | all | Rewriting + hybrid BM42, RRF 1:1, k = 60 |
| `F0`–`F3` | dev | Stage 2 fusion settings: RRF 1:1 k = 60; 3:1; 13:7; 1:1 k = 2 |
| `R1` | dev | Dense + fine-tuned IndoBERT reranker (score threshold 0.6) |
| `R2` | dev | F1 + fine-tuned IndoBERT reranker (score threshold 0.6) |
| `R3` | dev | F1 + generic mMiniLM reranker (no threshold) |
| `P0` | all | Full pipeline originally used by the chatbot: rewriting + BM42, RRF 13:7 with Qdrant's default k, IndoBERT reranker |
| `P1` | all | P0 with BM25 |
| `Clean-Baseline`, `Clean-Treatment` | in scope | Robustness check: the core pair on the corpus snapshot without duplicate uploads, exact vector search |

"dev" is the development subset used for stage 2 selection (38 items, 34 in scope; `dev_subset` in the test set). Comparisons with the baseline are paired by `item_id`. All arms share the same generator and prompt; see the paper §3.

## Fields

`data/testset.jsonl`: `item_id`, `question`, `reference_answer`, `validation_status`, `is_out_of_scope`, `excluded_after_run`, `in_analysis` (97 in-scope items used in the main analysis), `dev_subset`, `er5_evaluable` (source text recoverable from the raw corpus text; 93 items), `source_chunk_texts`, and `source_chunk_keys_at_run` (chunk identifiers of the source texts as they existed when the runs were made, one list per source text; used for Hit@5 and MRR).

`data/outputs/<arm>.jsonl`:

- `search_query`: query sent to the retriever (after rewriting where the arm uses it); `contexts`: retrieved units as sent to the generator; `contexts_meta`: chunk identifier, regulation type, number, year, article, validity flag, and retrieval or reranker score
- `answer`; `scores`: `context_recall`, `context_precision`, `answer_relevance`, `faithfulness` (judged against the generator context; `null` where not available), `factual_correctness`, on a 0 to 1 scale (the paper reports × 100)
- `hit_at_5`, `mrr`: rank of the first context whose chunk identifier matches a source chunk (`source_chunk_keys_at_run`)
- `evidence_retrieved_at_5`: every source text of the item is contained in the retrieved contexts (5-word shingle coverage ≥ 0.9), computed for every item with a source text, as in Table 10. Restricting to `er5_evaluable` items gives the rule of the companion chunking study.
- `refusal_detected`, `n_contexts`, `latency_s`

`real_queries/`: the 80 queries were sampled at random (seed 42) from 461 unique queries in the agent logs before 30 September 2026. The top-5 articles of eight arms were reconstructed without generating answers and pooled. Each pooled article was judged 0 (not relevant), 1 (supporting), or 2 (answers the question) by an LLM assessor (Gemini 3.5 Flash) for all queries, by Claude (Anthropic) for 10 queries, and by the first author for the same 10 queries through a blind annotation interface. `judge_in_scope_strict` is the assessor's scope decision under stricter instructions (paper §4.8). `rank_<arm>` is the article's position in that arm's top-5 (empty if absent). In `judgments.csv`, `article` identifies the regulation and article; the article texts are in the public regulations.

## Reproducing the main comparisons

```python
import pandas as pd
from scipy.stats import wilcoxon

s = pd.read_csv("data/scores.csv")
s = s[s.in_analysis]
w = s.pivot(index="item_id", columns="arm", values="context_precision") * 100
d = (w["Treatment"] - w["Baseline"]).dropna()
print(round(d.mean(), 2), wilcoxon(d).pvalue)   # -17.66, p < 0.001 (paper Table 10, main row)
```

## Corrections

**22 October 2026.** After the experiments, the reference answers of two test items were corrected because they cited provisions that had been amended or replaced:

- `7dc878fb-b988-4906-957c-cd6bb2026b54` (divorce or polygamy permits for civil servants who head a state-owned enterprise or state bank): the permit is granted by the President under Government Regulation 10/1983 Article 11(3) as amended by Government Regulation 45/1990, not by the technical minister.
- `f4396fc3-0a8c-450e-8362-d058fe3d0e64` (allowances for an employee transferred to an agency without sufficient budget): Sumenep Regent Regulation 6/2026 Article 28(2) places the allowance on the former agency until the new agency's budget is available.

`data/testset.jsonl` and all scores in this repository keep the **original** reference answers, because every reported result was computed with them. Excluding the two items (N = 95) leaves the direction and significance of every main comparison unchanged; see the paper's threats-to-validity section.

## Privacy and sources

- The corpus consists of 71 Indonesian civil-service regulations, which are public documents. Contexts are quoted from these regulations. Civil-servant ID numbers (NIP) appearing in regulation text are replaced with `[NIP]`.
- The text of real user queries is not released; only codes, lengths, scope decisions, and relevance judgments are included. No user identities are included. The test set contains no reviewer identities.

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See `LICENSE`. The license covers the dataset as compiled here. The regulation texts quoted in the contexts are official Indonesian government documents; laws and regulations (*peraturan perundang-undangan*) are not subject to copyright under Article 42 of Law No. 28 of 2014 on Copyright.

## Citation

[TODO: add the paper citation and, if deposited, the Zenodo DOI]

## Contact

Moh. Affan, b4affan@gmail.com (ORCID 0009-0009-2452-1437)
