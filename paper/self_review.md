# Neurocomputing (Elsevier, Q1) Self-Review & Quality Audit

**Manuscript:** *Explainable Deep Learning for Scene Classification and Pattern Recognition: A Cross-Paradigm Benchmark and Attribution-Consistent Adversarial Defense*  
**Authors:** MD Matiur Rahman ($^a$), Huanjie Tao ($^{b,*}$, Corresponding Author)  
**Journal:** *Neurocomputing* (`elsarticle.cls`, `5p,times,twocolumn`, Subscription Mode — \$0 APC)

---

## 1. Strict Verification Checklist

| Audit Item | Status | Verification Details |
| :--- | :---: | :--- |
| **Zero `[TBD]` Placeholders** | PASS | Verified programmatically (`TBD: []`). All 6 architectures (`ResNet-50`, `EfficientNet-B0`, `ConvNeXt-Tiny`, `Swin-Tiny`, `CLIP Zero-Shot`, `CLIP Linear`), `VG-LayerCAM`, `SAC-PGD-AT` ($\beta \in \{0.0, 0.05, 0.1, 0.2\}$), McNemar tests, and `SUN397-100` tables populated from `results/*.json`. |
| **Highlights Length ($\le 85$ chars)** | PASS | 5 bullets of lengths `[72, 73, 72, 71, 72]` characters (including spaces) in both `paper/main.tex` and `paper/highlights.md`. |
| **Graphical Abstract** | PASS | `figures/graphical_abstract.png` included inside `\begin{graphicalabstract}...\end{graphicalabstract}`. |
| **Abstract Word Count ($\le 250$ words)** | PASS | Verified `< 250` words, single paragraph, zero citations. |
| **Bibliography Integrity (`references.bib`)** | PASS | 48 real, verified references (all with DOI or arXiv ID), including `apedo2025crack`, `tao2024hierarchical`, `tao2023smoke`, `antamis2024interpretability`, `tang2024adversarial`, `sun2023multiobjective`, and `tulla2025phishing`. `48/48` cited in text. |
| **Tables & Figures Hygiene** | PASS | Every single-column `table` uses `\resizebox{\columnwidth}{!}{...}` and every two-column `table*` uses `\resizebox{\textwidth}{!}{...}`. All figures optimized so `paper_overleaf.zip` is `< 5 MB` for `<10s` Overleaf compilation. |
| **Elsevier Back-Matter Sections** | PASS | Includes CRediT authorship contribution statement, Declaration of competing interest, Data availability, and Acknowledgments. |
