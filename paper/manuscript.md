# Grouping duplicate citizen complaints in a state ombudsman: a first operational account from Ceará, Brazil

**Draft v0.5 — 24 September 2026** (canonical text: `paper/article.tex`)  
Practice paper (not yet submitted)

Francisco Nauber Bernardo Gois^{1,*} and Berg Silva^{1}

^{1} Controladoria e Ouvidoria Geral do Estado do Ceará (CGE-CE), Fortaleza, Brazil  
\* Corresponding author: francisco.gois@cge.ce.gov.br

Authorship is provisional. Other CGE staff who ran infrastructure, ETL, and Ouvidoria review should be added after the team agrees.

---

## Abstract

State ombudsman offices receive many complaints that tell the same story in different words. Officers then open several files for one event. Similaridade is the system used by the Controladoria e Ouvidoria Geral do Estado do Ceará (CGE-CE) to group those texts before a human links them. This draft reports the method that was in production in August 2026 and a homolog test of a later stack on the same 4,389 real complaints.

The production pipeline encodes each complaint with TF-IDF and groups neighbors with DBSCAN. A cluster is kept only when mean intra-cluster cosine similarity is at least 0.75. On that 4,389-complaint set the production stack grouped 62.4% of texts. A homolog stack that uses sentence embeddings, HDBSCAN, and a reasoning model on leftovers grouped 84.9% (3,725 complaints in 354 clusters). An LLM pass recovered 104 of 768 outliers into 21 themes. About 87% of the grouped items fell in a “green” band meant for bulk linking, which would leave officers with roughly 484 cases to inspect instead of 3,725.

The new stack is not a finished production claim. Short complaints still yield weak vectors. In September 2026 the live job at times collapsed to a single cluster. The Ouvidoria has not yet labeled a held-out sample. We write this as a practice note so the team can correct numbers, add authors, and decide whether to send a later version to *Digital Government: Research and Practice*.

**Keywords:** citizen complaints; ombudsman; text clustering; sentence embeddings; HDBSCAN; digital government; Brazil

---

## 1. Introduction

An ombudsman file starts as free text. The same broken pipe, the same delay at a public counter, or the same coordinated campaign can arrive tens of times in one morning. If each arrival becomes its own case, the officer repeats the same reading and the citizen waits longer.

Similaridade is CGE-CE’s answer to that pile. It runs on the complaint stream of the state Ouvidoria (COUVI). The job is narrow: put texts that likely refer to the same event in one group, mark the rest as leftovers, and show the officer a band (green, yellow, red) instead of a raw list.

This paper is a first English write-up of what the ASESI team already measured in August and September 2026. It is not a claim of a new clustering algorithm. TF-IDF, DBSCAN, sentence transformers, and HDBSCAN are known tools (Salton & Buckley, 1988; Ester et al., 1996; Reimers & Gurevych, 2019; Campello et al., 2013). The contribution, if the later version holds, is an operational account: one public-sector stream, two stacks on the same 4,389 complaints, and a triage rule that cuts the reading load.

Three questions guide the draft.

1. How much of a real complaint batch does lexical clustering already group?
2. How much more does a semantic stack group on the same batch?
3. What still breaks when the job runs every day?

---

## 2. Related work

Term weighting with TF-IDF remains a baseline for “same words, same document” (Salton & Buckley, 1988). It fails when two citizens describe the same event with no shared n-gram.

DBSCAN finds dense regions and treats sparse points as noise (Ester et al., 1996). That matches an ombudsman stream: some days one campaign dominates, other days many singleton complaints. The price is a fixed neighborhood radius. When density varies, one radius either glues unrelated texts or leaves paraphrases outside.

HDBSCAN drops the single radius and extracts clusters from a density hierarchy (Campello, Moulavi, & Sander, 2013). Sentence-BERT produces vectors that can be compared with cosine similarity without pairing every text through a cross-encoder (Reimers & Gurevych, 2019). That is the combination we tested in homolog: embed, group by hierarchical density, then ask a reasoning model only about leftovers.

We do not review the broader literature on public grievance portals here. A later revision should place Ceará next to other national complaint systems and say what is specific to Brazilian state ouvidorias.

---

## 3. Setting

CGE-CE is the state controller and ombudsman of Ceará. Similaridade serves the Ouvidoria. The named client group on 17 June 2026 was Otacílio, Bené, Jean, and Mara. ASESI builds and runs the software. Airflow starts the job. Counts later land in an `etl.similaridade_log` table (complaints processed, grouped, outliers, largest and smallest cluster, intra-cluster similarity).

The production schedule in August 2026 was three runs per day: 08:00, 12:00, and 17:00. Source code lives on the state’s GitLab (`analise_similaridade_dev`, branch `dev-api`). This public draft does not copy that repository.

Complaint text is sensitive. We report only aggregate counts already shared inside the team. No complaint body is reproduced.

---

## 4. Methods

### 4.1 Production stack (TF-IDF + DBSCAN)

Each complaint is a document. The encoder is TF-IDF over unigrams, bigrams, and trigrams, with the usual term frequency and smoothed inverse document frequency:

TF(t, d) = frequency of term t in document d

IDF(t) = ln[(1 + N) / (1 + df(t))] + 1

Vectors are L2-normalized. DBSCAN then groups points that share a neighborhood of a given radius. After clustering, a group is accepted only if mean pairwise cosine similarity inside the group is at least 0.75.

The team already recorded the main failure of that rule. Similarity is not transitive. A can sit close to B, B to C, and C to D, while A and D score near 0.006 and still share a label. On 25 August 2026 Berg Silva reported that pattern from production data.

A later check uses distance to the cluster centroid, expressed as a z-score, to flag “intruders” inside a group. That check behaved better on sentence embeddings than on TF-IDF (note from 27 August 2026).

### 4.2 Homolog stack (sentence embeddings + HDBSCAN + LLM)

The homolog pipeline has three steps.

1. Encode each complaint with a Sentence-Transformers model so paraphrase can share a vector neighborhood (Reimers & Gurevych, 2019).
2. Cluster with HDBSCAN, which does not need one global radius (Campello et al., 2013).
3. Send leftovers to DeepSeek R1. The model sees a reduced candidate list, not the full leftover set.

On a sample of about 4,000 complaints the embedding step finished in a little over three minutes on the machine Berg used for the test (25 August 2026). We do not yet publish the exact model name; that belongs in the next version after the GitLab note is copied with the team’s leave.

### 4.3 Operator bands

Grouped items are shown in three bands:

- green: high internal similarity, meant for bulk linking
- yellow: officer confirms
- red: officer reads with care

A cluster at 100% internal similarity means the texts are identical, not “almost the same.”

---

## 5. Results

### 5.1 Same batch, two stacks (n = 4,389)

The comparison below was reported to the ASESI group on 28 August 2026. Both rows use the same 4,389 real complaints from homolog.

| Stack | Share grouped | Extra detail |
| --- | --- | --- |
| Production: TF-IDF + DBSCAN | 62.4% | lexical near-copies |
| Homolog: Sentence-Transformers + HDBSCAN + DeepSeek R1 | 84.9% | 3,725 complaints in 354 clusters |

The leftover path on the homolog stack:

- 768 complaints left ungrouped
- 27 candidates passed to the LLM
- 104 complaints recovered
- 21 themes (examples named in the team note: prison, state water company, official vehicles)

About 87% of grouped items sat in the green band. Under that rule the officer would inspect on the order of 484 cases, not 3,725.

The absolute gain is 22.5 percentage points in share grouped (62.4% to 84.9%). That figure is a homolog measurement, not a production SLA.

### 5.2 One production day (28 August 2026)

The live TF-IDF job still ran that day. Berg’s execution note:

**08:00.** 49 complaints processed. 48 in one cluster (label 31). 1 outlier. Mean intra-cluster similarity 85.68%.

**12:00.** 78 complaints processed. 76 in two clusters (labels 32 and 33). 2 outliers. Largest cluster: 73 members at 100% similarity. Smaller cluster: 3 members at 85.76%. Mean similarity across the two groups: 92.88%.

**Day.** 127 processed, 124 grouped (97.6%), 3 outliers.

A 97.6% grouping rate on a campaign day is not in conflict with the 62.4% rate on the 4,389-complaint set. The long set mixes ordinary days and waves. 28 August was a wave day: one cluster of 48 and one of 73 read as copy-paste or light paraphrase.

### 5.3 What broke after the test

September 2026 showed the gap between a homolog notebook and a daily Airflow job.

- 2 September: a homolog build of the new stack failed, then came back later the same day.
- 4 September: the live grouping collapsed toward one cluster and many outliers. Infrastructure blocked Python library downloads.
- 8 September: the Airflow worker loaded Sentence-Transformers 5.1.2 against a model saved with 5.4.0.
- 16 September: the system was in production but “strained.” Short texts carry little signal. Cosine grouping then fails.

So the 84.9% figure is a batch result. It is not yet the steady daily rate.

---

## 6. Discussion

Lexical clustering already catches the easy wave: identical or near-identical paste. That is useful on a morning like 28 August. It is the wrong tool when two people write the same event in different words.

The homolog stack moves the cut from “same tokens” to “same sense,” then spends the LLM only on leftovers. That design is cheap relative to sending every pair to a chat model. The 768 → 27 → 104 funnel is the part a later paper should stress-test. If the 27 candidates were chosen by a heuristic that already knew the 104, the LLM credit is overstated. The team note does not spell out that filter. Version 0.2 must.

The green band is a product decision, not a statistical guarantee. Without COUVI labels we cannot report precision of automatic linking. A cluster of 73 identical texts is easy. A cluster of paraphrases at 0.76 mean similarity is the case that will decide whether officers trust the button.

Short complaints remain the hard case. An embedding of a ten-word sentence is a poor object for cosine grouping. A later version should split the stream: very short texts go to a different rule (same agency, same day, shared unique token) instead of the same HDBSCAN run.

---

## 7. Limitations

This is a first draft. The limits are part of the result.

1. Numbers come from team reports of homolog and production runs, not from a frozen public dataset.
2. COUVI has not labeled a sample. There is no precision, recall, or agreement score.
3. The exact sentence-transformer checkpoint is not in this file.
4. DeepSeek R1 is a moving model. A paper that names it must pin a date and a prompt.
5. September failures (one-cluster collapse, library versions, short text) show that the 84.9% run is not the current daily truth.
6. We do not measure days saved for the citizen or minutes saved for the officer.
7. Authorship is incomplete.

None of these points is cosmetic. A submission without (2) and (6) will read as a vendor slide.

---

## 8. Ethics and data

Complaint text can identify a person or a public servant. This draft uses only counts. Any later replication package must stay inside CGE, with access logged, or use a redacted sample approved by the Ouvidoria. Clustering is an aid. An officer still closes the case.

---

## 9. Conclusion

On 4,389 real complaints, the production TF-IDF + DBSCAN job grouped 62.4% of texts. A homolog job with sentence embeddings, HDBSCAN, and a leftover LLM grouped 84.9% and recovered 104 outliers. A single production morning in August showed how lexical clustering behaves when a wave of identical texts arrives: 97.6% grouped that day, almost all in two large clusters.

The next version of this paper needs three things the first version cannot invent: COUVI labels on a sample of green and yellow clusters, a pinned embedding model, and a week of `etl.similaridade_log` after the Airflow environment is stable. Until then the honest claim is smaller. The Ouvidoria already groups copy-paste waves. A semantic stack grouped more of the same historical batch. Daily production of that stack is still being earned.

---

## Acknowledgments

ASESI colleagues who discussed the method in June–September 2026: Ana Luiza Cruz, Charles, Leonardo Borba, Oton, and the COUVI officers named in Section 3. Lucas Pimentel hosts the internal GitLab project. Infrastructure notes came from the ASESI INFRA and ASESI DEV groups. Errors in this draft are the authors’.

---

## CRediT (provisional)

- Conceptualization, writing (original draft): Francisco Nauber Bernardo Gois
- Methodology, software, investigation (homolog measurements): Berg Silva
- Other roles (data curation, infrastructure, validation by Ouvidoria): to be assigned

---

## Funding

No external grant. Work done as part of CGE-CE / ASESI operations.

---

## Competing interests

The authors work for the agency that operates the system. That is the point of a practice paper. It is also a conflict to declare.

---

## Data availability

Complaint text is not public. Aggregate figures in Section 5 can be checked against CGE internal logs by staff with access. A redacted sample would need Ouvidoria approval.

---

## Use of AI tools

A language model helped organize this first English draft from team notes. The counts and method steps were taken from those notes, not generated. The corresponding author is responsible for every number.

---

## References

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In J. Pei, V. S. Tseng, L. Cao, H. Motoda, & G. Xu (Eds.), *Advances in Knowledge Discovery and Data Mining* (PAKDD 2013, LNCS 7819, pp. 160–172). Springer. https://doi.org/10.1007/978-3-642-37456-2_14

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining* (KDD-96, pp. 226–231). AAAI Press. https://cdn.aaai.org/KDD/1996/KDD96-037.pdf

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
