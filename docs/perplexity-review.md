# Perplexity review — Similaridade v0.4

- Date: 23 September 2026
- Provider: `openrouter` / `perplexity/sonar-pro`
- Source: `paper/article.tex` (v0.4)
- Score history: v0.3 first pass 3.0 → v0.3 after counting table 2.9 → **v0.4 = 3.1/5**
- Honesty rose to 5. Reproducibility fell to 1 because the manifest now lists the holes instead of hiding them. SOTA rose to 3 after Rabbi et al.\ (JIDM 2026) and Silva et al.\ (PROPOR 2026).

Search-backed editorial read. Not a human associate editor. Sonar reads stripped TeX; the PDF has a complete bibliography.

## Verdict
Revise for DGOV practice paper.

## Score

| Criterion | Score (0–5) |
|---|---:|
| Fit to DGOV practice track | 4 |
| Honesty of claims | 5 |
| Method reproducibility | 1 |
| Evidence quality | 2 |
| State of the art coverage | 3 |
| Operational usefulness | 4 |
| Ethics and data handling | 3 |
| **Mean** | **3.1/5** |

## What holds

- The manuscript makes a **narrow, operationally credible claim**: lexical clustering can group copy-paste complaint waves.
- It clearly distinguishes **grouping rate from precision**, and explicitly states that no officer gold standard exists.
- It separates the reported 84.9% result from the better-supported HDBSCAN-only inference of 82.5%.
- The production setting is concrete: CGE-CE, COUVI, Airflow, three daily runs, an internal log table, and a documented September failure.
- The manuscript treats the language-model pass appropriately as an **undocumented ablation**, not as a deployment success.
- The proposed human-validation protocol is directionally suitable for an applied government paper.

## What blocks submission

1. **The central comparison is not reproducible.**  
   The encoder checkpoint, DBSCAN radius, HDBSCAN parameters, distance metric, preprocessing, random seed, software versions, and acceptance rules are unknown. This matters because the paper’s main result is a comparison between algorithms, but the reader cannot rerun either treatment.  
   **Smallest fix:** freeze one rerun of the same 4,389 records and provide a complete machine-readable manifest: text preprocessing, model identifier and hash, all clustering parameters, software versions, random seeds, cluster post-processing, and exact counts.

2. **The 82.5% result is derived rather than directly observed.**  
   The manuscript computes \(3{,}725-104=3{,}621\), but the conservation table does not establish that every recovered item was previously included in the 768 leftovers or that the treatments used identical input and post-processing rules.  
   **Smallest fix:** publish a row-level accounting table inside CGE, or an auditable aggregate ledger with mutually exclusive counts and checksums showing treatment membership before and after each step.

3. **The 27-to-104 funnel is internally unresolved.**  
   Twenty-seven candidates cannot straightforwardly yield 104 recovered complaints without a seed-expansion, bundle, or compressed-list rule. This is a material threat to the claimed LLM increment and may indicate leakage or circular candidate selection.  
   **Smallest fix:** specify the candidate-generation algorithm and disclose whether candidate selection used labels, cluster membership, human inspection, or the eventual LLM output; then rerun the ablation with a pre-registered candidate rule.

4. **There is no outcome validation.**  
   “Grouped” currently means “placed in an accepted cluster,” not “same event.” A higher grouping rate may therefore represent over-merging, including merging unrelated victims or complaints about the same theme. This is the paper’s principal validity gap.  
   **Smallest fix:** label a stratified sample for same event, related theme, different, and unclear, with blinded dual officers and adjudication; report pairwise precision, false-merge rate, false-split rate, and BCubed precision/recall.

5. **The unit of deduplication is underspecified.**  
   The manuscript alternates among same complaint, same event, campaign, theme, and duplicate consumer submission. These are not equivalent operational targets.  
   **Smallest fix:** choose one primary target—preferably “same event suitable for one case file”—and report theme/campaign grouping separately as a different task.

6. **The production claim is too weakly instrumented.**  
   One campaign day is useful evidence, but the September collapse shows that the homolog stack was not operationally stable. The paper lacks a multi-day denominator, run-level failure rate, drift monitoring, and alerting policy.  
   **Smallest fix:** add at least one frozen week of run logs after the environment is pinned, including processed count, grouped count, leftovers, cluster-size distribution, model/version identifiers, runtime, errors, and anomaly alerts.

7. **The data-protection analysis is incomplete for a government deployment.**  
   The manuscript correctly flags uncertainty about whether DeepSeek R1 received sensitive text, but it does not identify the legal/organizational basis, data classification, access controls, retention, provider-training terms, or incident process.  
   **Smallest fix:** add a short data-flow and governance table covering storage, processing location, access, retention, deletion, provider terms, audit logs, and the decision to prohibit external LLM processing until approval.

8. **The literature review contains unverifiable or insufficiently specified citations.**  
   Several keys are not named in the manuscript’s reference list, and at least one cited item appears to be a 2026 work whose bibliographic details need verification. A practice paper cannot rely on placeholder keys or citations that a reviewer cannot identify.  
   **Smallest fix:** replace every key with a complete reference containing authors, title, year, venue, and DOI or stable public record; remove any item that cannot be independently verified.

9. **The manuscript is not yet a finished submission.**  
   It contains provisional authorship, “not submitted,” conceptual figures generated by Grok Imagine, broken LaTeX/entities, missing references, incomplete CRediT text, and several unresolved table labels.  
   **Smallest fix:** produce a clean v0.4 PDF/source package, complete authorship and references, replace decorative figures with one reproducible pipeline figure and one empirical results figure, and remove drafting artifacts.

## Missing literature

- **“Detecting and Analysing Duplicate Consumer Complaints and Collective Demands” (2026, Journal of Information and Data Management).**  
  This is the closest public comparison identified in the search material. It combines duplicate detection across consumer-complaint platforms with semantic clustering of collective demands, and reports use of temporal and cross-platform attributes. The manuscript should distinguish its “same consumer duplicate” task from Ceara’s “same event” task and discuss metadata such as identity, provider, subject, and time window. *Bibliographic details should be checked against the final journal record before citation.*

- **“Classification of Public Administration Complaints” (2022, SLATE—International Conference on Source Code Analysis and Manipulation? / OASIcs record).**  
  This is relevant public-administration complaint NLP, particularly for positioning the work against routing/classification rather than deduplication. The authors should use it to make the task distinction more precise and to report why unsupervised grouping is operationally preferable here. *The exact conference expansion and final bibliographic metadata should be verified.*

- **“Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks” (2019, EMNLP-IJCNLP).**  
  The manuscript cites this method but should state whether the deployed checkpoint is multilingual, Portuguese-specific, or domain-adapted. The paper is foundational for the embedding choice, but it does not justify the particular model or threshold.

- **“BERTimbau: pretrained BERT models for Brazilian Portuguese” (2020, Brazilian Conference on Intelligent Systems—BRACIS).**  
  This is directly relevant to Portuguese-language representation and should be used to motivate, test, or reject a Brazilian-Portuguese encoder. The manuscript currently mentions Portuguese checkpoints without selecting or evaluating one.

- **“Density-Based Spatial Clustering of Applications with Noise” (1996, KDD).**  
  The DBSCAN citation is appropriate, but the paper should explain how the radius and minimum-points parameters were selected and how noise is counted. A method citation alone cannot substitute for an experiment manifest.

- **“A Generalized HDBSCAN Algorithm for Clustering of High-Dimensional Data” (2013, Advances in Knowledge Discovery and Data Mining).**  
  This supports the HDBSCAN rationale, but the manuscript should report whether dimensionality reduction was used and whether clustering occurred directly in embedding space. The distinction is important because UMAP-plus-HDBSCAN is common in topic modeling but is not the same pipeline.

- **“BERTopic: Neural topic modeling with a class-based TF-IDF procedure” (2022, arXiv).**  
  The manuscript correctly distinguishes topic discovery from same-event grouping. It should nevertheless cite this work with complete metadata and explain whether its cluster-labeling and topic-coherence practices informed the proposed leftover “theme” terminology.

- **“BCubed: A Formal Definition and Evaluation of Clustering Quality” (2009, Journal of Machine Learning Research).**  
  This is the appropriate basis for cluster-quality evaluation once officer labels exist. The paper should define how “unclear” labels and adjudicated clusters will be handled.

- **“On the Resemblance and Containment of Documents” (1997, Compression and Complexity of Sequences).**  
  This is useful historical background for near-duplicate detection, but it should not carry the burden of public-sector validity. Add contemporary complaint-specific evidence and explain what “same event” adds beyond textual resemblance.

- **Current public-sector LLM grievance-classification work identified in the 2025–2026 literature search.**  
  The manuscript should review recent work on LLM-assisted routing and grievance analysis only as adjacent evidence, not as direct validation of deduplication. In particular, the authors should separate classifier accuracy from cluster validity, and vendor/system demonstrations from peer-reviewed evidence. Any 2026 preprint or workshop item should be marked as such and not presented as established state of the art.

## Numbers to pin

- Exact extraction window for the 4,389 complaints.
- Dataset checksum or immutable batch identifier.
- Number of complaints with empty, extremely short, or non-Portuguese text.
- Text-length distribution: token count percentiles and number below 20 tokens.
- Agency, service, channel, geographic, and time-window distributions.
- Exact TF-IDF vocabulary and preprocessing: casing, accents, stop words, stemming, and n-gram boundaries.
- Exact DBSCAN radius, `min_samples`, metric, and treatment of noise.
- Exact HDBSCAN metric, `min_cluster_size`, `min_samples`, cluster-selection method, and any dimensionality reduction.
- Number of clusters, singleton/noise items, and cluster-size histogram for every treatment.
- Definition and computation of “mean intra-cluster cosine similarity,” including whether singleton clusters are eligible.
- Number of pairwise comparisons used for each accepted cluster.
- The exact source of the 0.75 acceptance threshold and whether it was tuned on this batch.
- Number of clusters and complaints in each green, yellow, and red band.
- Exact band thresholds and whether they were chosen before seeing outcomes.
- The 768 leftovers: their complete disposition after the HDBSCAN stage.
- The 27 candidates: selection rule, candidate-to-complaint mapping, and whether candidates were seeds or individual texts.
- The 104 recoveries: count of complaints, clusters, and unique seed candidates.
- The 21 leftover groups: whether they are inside or outside the 354 HDBSCAN clusters.
- Prompt, model version, decoding parameters, context limits, and output parser for DeepSeek R1.
- Hosting location, data egress, retention, access logs, and provider-training terms for the LLM call.
- Per-treatment officer precision, recall, false-merge rate, false-split rate, BCubed metrics, confidence intervals, and inter-rater agreement.
- Daily production metrics over at least one frozen week: failures, runtime, grouped rate, leftovers, largest cluster, and model/software versions.
- Number of officer accepts, rejects, splits, and minutes per 100 complaints in the live queue.
- Arithmetic reconciliation for the reported 2,739 production grouped complaints and all implied counts.

## Suggested next version

1. Replace the current inferred comparison with a frozen, reproducible experiment manifest and rerun all three treatments on the identical batch; include exact parameters, model hashes, software environment, random seeds, and row-level or auditable aggregate conservation.

2. Define “same event” as the primary label and separate it from “same consumer duplicate,” “related theme,” and “coordinated campaign”; revise every result, table, and figure to use those terms consistently.

3. Run the proposed blinded officer study before making any semantic-improvement claim; report treatment-level precision, false merges, false splits, BCubed metrics, agreement, and uncertainty by cluster-size and text-length stratum.

4. Remove or quarantine the LLM result unless the 27-candidate rule, 104-recovery accounting, prompt, model version, and data-flow approval are documented; if retained, present it as a separately evaluated exploratory treatment.

5. Add a production-reliability section based on a frozen week of logs, including the September failure modes, dependency pinning, model compatibility checks, drift/anomaly alerts, rollback behavior, and a human-review safeguard that prevents automatic case consolidation when confidence or infrastructure checks fail.