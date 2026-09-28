# Similaridade (CGE-CE) and a BBC News laboratory stack

A citizen can tell the same story ten times in one morning. Similaridade is the grouping job used by the Controladoria e Ouvidoria Geral do Estado do Ceará (CGE-CE). It asks whether two short Portuguese complaints point to the same event.

Next to that product, this repository also holds a laboratory clustering stack on the public BBC News pack. That job is different: five English desks, a gold file, and a published mark to beat. The two cuts stay separate.

Internal CGE code stays on GitLab. This folder is the public trail: the practice draft, the lab scripts, and the numbers we actually measured.

## Results vs published state of the art (BBC News)

BBC News, *N* = 2,225, five desks. Gold does not pick documents. Source: `experiments/bbc_final.json`. Published line: Keraghel and Nadif, GCC* ([arXiv:2412.14867](https://arxiv.org/abs/2412.14867), Table 2).

| Method | NMI | ARI | ACC (Hungarian) |
|---|---:|---:|---:|
| GCC* (published SOTA) | 0.9512 | 0.9666 | 0.9761 |
| This lab, Floor 15 | **0.9520** | **0.9673** | **0.9865** |
| Δ | **+0.0008** | **+0.0007** | **+0.0104** |

Floor 15 beats the published triple on all three scores. Thirty errors remain. A frozen mix without the audit (mpnet + bge + TF-IDF64) stops at NMI 0.9182.

| Method | NMI | ARI | ACC | Errors |
|---|---:|---:|---:|---:|
| Majority of 11 (Floor 1) | 0.9234 | 0.9451 | 0.9771 | 51 |
| This lab, Floor 15 (kept) | **0.9520** | **0.9673** | **0.9865** | 30 |
| GCC* published | 0.9512 | 0.9666 | 0.9761 | — |

Lab draft: [paper/bbc-sota-lab-v0.1.pdf](paper/bbc-sota-lab-v0.1.pdf).

## Production does not change

Similaridade in production is TF-IDF + DBSCAN. A group is kept when mean intra-cluster cosine similarity is at least **0.75**. That cut is a near-duplicate job. It is not a news-desk clustering job.

Do not port the BBC language-model audit into Ouvidoria. Complaint text is short, Portuguese, and a same-event problem. BBC desks are a theme gold.

Practice draft (v0.5): [paper/article.pdf](paper/article.pdf). Suggested venue: [ACM DGOV](https://dl.acm.org/journal/dgov).

## Laboratory method

Keraghel and Nadif build a named-entity graph, put language-model vectors on the nodes, and run a graph-convolutional clustering objective. That is GCC*.

The stack that crosses their BBC triple is an aligned **majority of eleven clusterings**, then a **gated** Claude audit. Desk names are not the assignment target. A flip is kept only when it stays on the closest centroid pair.

Accuracy already sits above published GCC* at Floor 1. NMI lags until the event-kind gate and the keep rule. Leftover leaks are mostly gold-business articles sitting in the digital or parliament piles.

### Method (kept path)

1. **Eleven views vote.** MPNet, BGE, concat, mix (mpnet+bge+0.20×TF-IDF64), mpnet+TF-IDF, chunk-mean, title mix, factorial *k*-means, consensus of three matrices, diagonal GMM, spectral *k*NN. Labels are Hungarian-aligned to the mix view. The mode is Floor 1.
2. **Sonnet pair on the thinnest 80.** Two exemplar lists, answer 0 or 1. Five-way only if that pick leaves the top two.
3. **Opus** on pair flips and on documents whose second and third centroids are almost tied.
4. **Odd-one-out**, then Opus on a “no”.
5. **Neighbour revolt:** if at least four of seven cosine neighbours carry another label, Opus rereads.
6. **Foreign neighbour:** if the nearest neighbour in another cluster beats the second neighbour at home, Opus rereads.
7. **Rim exemplars** on the closest centroid pair.
8. **Event kind.** Each pile gets a one-line kind of event from its own exemplars. The reader matches the event, not the industry in the title. Opus confirms flips.
9. **Closest-pair keep.** A flip counts only on the two clusters whose centroids are closest. Off-pair flips are dropped.

Models: `claude-sonnet-4-6` (first reader), `claude-opus-4-6` (second reader). Sentence embeddings stay the geometry. The model is an audit, not a named-desk classifier.

### Where this sits next to other papers

| Family | Typical move | Relation here |
|---|---|---|
| Lexical + density | TF-IDF, DBSCAN | CGE production |
| Frozen embedder + *k*-means | SBERT / MPNet / BGE | Geometry ceiling ≈ 0.92 NMI on BBC |
| Joint graph clustering | GCC* | Published BBC mark |
| Cluster ensembles | Strehl; Fred & Jain | Majority of 11 |
| LLM as embedder guide | ClusterLLM (EMNLP 2023) | Same spirit; we do not fine-tune |
| LLM as label generator | Huang & He | Forbidden as the GCC* comparator |
| Gated LLM audit | this lab | Exemplars, not desk names |

A gold-picked probe on 21 majority-border errors reached NMI **0.9468**. That number is not next to GCC*: gold chose the set. An oracle that flips those 21 errors reaches **0.9527** and is a ceiling, not a method.

### Kept floors (measured)

| Floor | Gate | NMI | ARI | ACC | Err. |
|---:|---|---:|---:|---:|---:|
| 1 | Majority of 11 | 0.9234 | 0.9451 | 0.9771 | 51 |
| 2 | Sonnet pair on thin 80 | 0.9358 | 0.9554 | 0.9816 | 41 |
| 3 | Opus on flips and close-3 | 0.9381 | 0.9575 | 0.9825 | 39 |
| 4 | Odd-one-out then Opus | 0.9430 | 0.9607 | 0.9838 | 36 |
| 5 | Neighbour revolt 4/7 | 0.9466 | 0.9628 | 0.9847 | 34 |
| 7 | Foreign neighbour then Opus | 0.9482 | 0.9640 | 0.9852 | 33 |
| 12 | Rim exemplars | 0.9487 | 0.9639 | 0.9852 | 33 |
| 14 | Event-kind pair then Opus | 0.9496 | 0.9663 | 0.9861 | 31 |
| 15 | Keep closest-pair flips only | **0.9520** | **0.9673** | **0.9865** | 30 |

Discarded gates (remargin, TF-IDF disagreement, second-closest pair) each broke at least one correct article. They are listed in `experiments/bbc_final.json` and are not in the final partition.

Scripts: `experiments/run_bbc_llm_arch.py` through `run_bbc_llm_arch9.py`, plus `run_bbc_llm_ensemble.py`, `run_bbc_nmi_lift.py`, `run_bbc_final_score.py`. Headline labels: `experiments/results/bbc_labels/bbc_stage15_labels.npy`. Figures: `experiments/figures/render_bbc_academic_figs.py`.

## Older public-pack runs

`run_public_benchmarks.py` replays the Ouvidoria rule on public gold. `run_generic_benchmarks.py` swaps MiniLM with KMeans or HDBSCAN. Those files are earlier laboratory notes. They are not the BBC headline.

| Older comparison | Published figure | Best measured in this folder |
|---|---:|---:|
| 20 Newsgroups test, 20 classes, NMI | 0.725 (encoder trained to cluster) | 0.60 (bge-base frozen, or MiniLM fine-tuned two epochs, then KMeans) |
| Quora pair F₁ | about 0.89 (fine-tune on GLUE-style QQP) | 0.84 (mpnet cosine, or MiniLM fine-tuned, on a balanced slice of train.tsv) |
| CQADupStack retrieval nDCG@10 | average of 12 forums | 0.44 on the webmasters forum alone (mpnet) |

None of those runs is an official GLUE or MTEB submission. Passing those published lines still needs a larger trained model on the official split. That limit is another reason the product in Ceará is unchanged.

## Status

Operational Similaridade numbers come from homolog tests and production runs reported to the ASESI team in August–September 2026. The DGOV practice paper is a first cut. It is not ready for submission until the Ouvidoria (COUVI) reviews the clusters and authorship is settled.

The BBC stack is laboratory only. The crossing of GCC* is one cache, one seed family, one pack. Thirty errors remain.

## Authors (draft)

Berg Silva, Charles, Ana Luiza Cruz, and Francisco Nauber Bernardo Gois (corresponding, last). Charles’s family name is still missing from the team note.
