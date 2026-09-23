# Sources for v0.1

Facts below were taken from ASESI team chats (ASESI, ASESI DEV, ASESI SIEC, ASESI INFRA, ASESI IA) and direct messages with Berg Silva and Oton, June–September 2026. No citizen names, no phone numbers, no complaint text.

## System and client

- Client area: Ouvidoria / COUVI (Otacílio, Bené, Jean, Mara), stated in ASESI SIEC on 17 June 2026
- Code: internal GitLab `analise_similaridade_dev`, branch `dev-api`
- Orchestration: Airflow ETL
- Ticket mentioned in Trello update: 203528
- CONACI inventory (9 September 2026): Similaridade listed as grouping similar complaints; ask was to mark status as production
- Ceará Íntegro 2026 medal inscription discussed in ASESI (1–9 September)

## Production method (in use in August 2026)

- TF-IDF on unigrams, bigrams, trigrams
- Density clustering (DBSCAN) with a neighborhood radius
- A group is kept only if mean intra-cluster similarity ≥ 0.75
- Known flaw (Berg, 25 August): A similar to B, B to C, C to D, yet A far from D; pairwise scores as low as 0.006 inside a cluster
- Three daily runs: 08:00, 12:00, 17:00
- Metrics: mean intra-cluster similarity; overall similarity as the mean across formed groups
- 100% similar means the texts are identical
- Logs later written to `etl.similaridade_log` (counts processed, grouped, outliers, largest and smallest cluster)

## Production day 28 August 2026 (Berg report)

- 08:00: 49 complaints, 48 in one cluster (label 31), 1 outlier, mean similarity 85.68%
- 12:00: 78 complaints, 76 in two clusters (labels 32 and 33), 2 outliers; largest cluster 73 members at 100%; smaller cluster 3 members at 85.76%; overall similarity 92.88%
- Day total: 127 processed, 124 grouped (97.6%), 3 outliers
- Large clusters read as copy-paste or light paraphrase waves

## Homolog comparison on 4,389 real complaints (28 August 2026)

Reported to ASESI the same evening:

| Pipeline | Grouped | Detail |
|---|---|---|
| Production TF-IDF + DBSCAN | 62.4% | lexical near-duplicates |
| Sentence-Transformers + HDBSCAN + DeepSeek R1 | 84.9% | 3,725 complaints in 354 clusters |

- Outlier path: 768 ungrouped → 27 candidates → DeepSeek R1 → 104 complaints recovered in 21 themes
- Operator bands: green (bulk link), yellow (confirm), red (close review)
- About 87% of grouped items in green; operator would review about 484 cases, not 3,725
- Sentence-Transformers on about 4,000 complaints: a little over 3 minutes (Berg, 25 August)
- Hybrid proposal: TF-IDF + DBSCAN **or** Sentence-Transformers + HDBSCAN (better overall) and an LLM on leftovers
- Intruder check: centroid distance with z-score, better with sentence embeddings (Berg, 27 August)
- COUVI review still required before calling the new method production

## Later operations (September 2026)

- 2 September: homolog of the new stack failed; later the same day Berg said similaridade was working
- 4 September: process collapsing to one cluster and many outliers; infra blocking Python library downloads
- 8 September: Sentence-Transformers version clash on the Airflow worker (library 5.1.2, model saved with 5.4.0)
- 10 September: homolog environment updated (Leonardo)
- 16 September (ASESI DEV): in production but strained; short texts give poor embeddings; cosine grouping fails on those cases

## What this draft does not have

- Official export from `etl.similaridade_log`
- Human labels from COUVI
- Inter-rater agreement
- Wall-clock time saved per officer
- The exact sentence-transformer model name
