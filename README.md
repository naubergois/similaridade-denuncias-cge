# Similaridade — grouping similar ombudsman complaints (CGE/CE)

First English draft of a practice paper on the Similaridade system used by the Controladoria e Ouvidoria Geral do Estado do Ceará (CGE-CE). The pipeline groups duplicate and near-duplicate citizen complaints so the Ouvidoria does not treat the same case many times.

This folder is separate from other ASESI repos. Internal code stays on CGE GitLab.

## Draft (v0.5)

- [paper/article.pdf](paper/article.pdf) — ACM DGOV look (`acmlarge`), the file to read
- [paper/article.tex](paper/article.tex) — source in the official ACM template
- [paper/article-plain.pdf](paper/article-plain.pdf) — same text, simpler layout
- [paper/figures/](paper/figures/) — Grok Imagine conceptual figures plus matplotlib charts
- [docs/perplexity-review.md](docs/perplexity-review.md) — Perplexity Sonar evaluation
- [docs/journal.md](docs/journal.md) — venue and format notes
- [docs/sources.md](docs/sources.md) — team notes used (no personal data)

Format for review at ACM is `\documentclass[manuscript]{acmart}`. The compiled PDF uses `acmlarge`, which is the DGOV published page.

## Suggested journal

**ACM Digital Government: Research and Practice (DGOV)**  
https://dl.acm.org/journal/dgov

It publishes applied public-sector work and has a practice track. Government Information Quarterly is a later stretch if the COUVI validation and a longer time series land.

## Status of this version

Operational numbers come from homolog tests and production runs reported to the ASESI team in August–September 2026. The paper is a first cut. It is not ready for submission until the Ouvidoria (COUVI) reviews the clusters and authorship is settled.

## Authors (draft)

To be confirmed with the team. Working list: Francisco Nauber Bernardo Gois (corresponding), Berg Silva, and other CGE contributors named in acknowledgments.
