# Similaridade — grouping similar ombudsman complaints (CGE/CE)

First English draft of a practice paper on the Similaridade system used by the Controladoria e Ouvidoria Geral do Estado do Ceará (CGE-CE). The pipeline groups duplicate and near-duplicate citizen complaints so the Ouvidoria does not treat the same case many times.

This folder is separate from other ASESI repos. Internal code stays on CGE GitLab.

## Draft (v0.5)

- [paper/article.pdf](paper/article.pdf) — ACM DGOV look (`acmlarge`), the file to read
- [paper/article.tex](paper/article.tex) — source in the official ACM template
- [paper/article-plain.pdf](paper/article-plain.pdf) — same text, simpler layout
- [paper/figures/](paper/figures/) — Grok Imagine scenes with rates and labels typeset from team counts; 28 August chart in matplotlib
- [docs/perplexity-review.md](docs/perplexity-review.md) — Perplexity Sonar evaluation
- [docs/journal.md](docs/journal.md) — venue and format notes
- [docs/sources.md](docs/sources.md) — team notes used (no personal data)
- [experiments/](experiments/) — public packs. The Ouvidoria product stays TF-IDF + DBSCAN at cosine 0.75. The scripts below are laboratory only. Do not port them into CGE production.

Format for review at ACM is `\documentclass[manuscript]{acmart}`. The compiled PDF uses `acmlarge`, which is the DGOV published page.

## Suggested journal

**ACM Digital Government: Research and Practice (DGOV)**  
https://dl.acm.org/journal/dgov

It publishes applied public-sector work and has a practice track. Government Information Quarterly is a later stretch if the COUVI validation and a longer time series land.

## Public packs, measured here

A cosine of 0.75 catches near-copies. It does not rebuild a newsgroup, a news desk, or a loose chain of paraphrases. That split is why the production job stays as it is, and why the laboratory uses a different stack on public gold.

`run_public_benchmarks.py` replays the Ouvidoria rule. `run_generic_benchmarks.py` swaps in MiniLM with KMeans for themes and HDBSCAN for duplicate questions. On those packs the generic BCubed F₁ rises (BBC 0.01 to 0.92, Quora 0.13 to 0.46). That rise is against the CGE rule, on someone else's gold.

`run_sota_alternatives.py` then tries stronger frozen encoders (mpnet, bge-base), another clusterer, the official-shaped scores. `run_trained_alternatives.py` fine-tunes MiniLM and adds BM25. Numbers are written only after the run, in the JSON next to each script.

| Published comparison | Published figure | Best measured in this folder |
|---|---:|---:|
| 20 Newsgroups test, 20 classes, NMI | 0.725 (encoder trained to cluster) | 0.60 (bge-base frozen, or MiniLM fine-tuned two epochs, then KMeans) |
| Quora pair F₁ | about 0.89 (fine-tune on GLUE-style QQP) | 0.84 (mpnet cosine, or MiniLM fine-tuned, on a balanced slice of train.tsv) |
| CQADupStack retrieval nDCG@10 | average of 12 forums | 0.44 on the webmasters forum alone (mpnet). BM25 was 0.32. A half-and-half mix was 0.40 |

Agglomerative clustering lost on every theme pack. Four epochs of a MiniLM cross-encoder on a balanced Quora slice stayed at F₁ 0.84 while the fit loss kept falling: the small model memorizes the slice and does not reach the published 0.89. None of these runs is an official GLUE or MTEB submission. Passing 0.725 or 0.89 still needs a larger trained model on the official split. This limit is the reason the product in Ceará is unchanged.

## Status of this version

Operational numbers come from homolog tests and production runs reported to the ASESI team in August–September 2026. The paper is a first cut. It is not ready for submission until the Ouvidoria (COUVI) reviews the clusters and authorship is settled.

## Authors (draft)

Berg Silva, Charles, Ana Luiza Cruz, and Francisco Nauber Bernardo Gois (corresponding, last). Charles’s family name is still missing from the team note.
