# IEEE Access Self-Review & Quality Audit

**Manuscript:** *Accuracy, Faithfulness, and Robustness: An Explainable Benchmark of CNNs, Vision Transformers, and Vision-Language Models for Scene Recognition*  
**Date:** 2026-11-19

---

## 1. Strict Verification Checklist

| Audit Item | Status | Verification Details |
| :--- | :---: | :--- |
| **Zero `[TBD]` Placeholders** | PASS | Verified programmatically (`TBD: []`). All Intel 3-seed, CLIP, PGD-AT, McNemar, and SUN397-100 tables populated. |
| **Numeric Fidelity to `results/`** | PASS | Every number in Abstract, Tables I–VII, and Results prose cross-checked against `results/multi_seed_intel.json`, `results/mcnemar_intel.json`, `results/pgd_at_intel.json`, `results/multi_seed_sun397_100.json`, and `results/mcnemar_sun397_100.json`. |
| **Bibliography Integrity (`references.bib`)** | PASS | 42 real references; 0 missing citations; every BibTeX entry includes a verified DOI or arXiv identifier. `tulla2025phishing` cited accurately as collaborative prior work (5th author). |
| **Figures & Cross-References** | PASS | All 16 included `.png` figures verified present in `paper/figures/` at 300 DPI; 0 undefined `\ref{}` labels. |
| **Author & Co-Author Block** | PASS | Lists only `MD Matiur Rahman` with `% TODO: co-author permission (Prof. Huanjie Tao)` preserved. |
| **LaTeX Markup Hygiene** | PASS | Fixed a Markdown bold artifact (`**...**` -> `\textbf{...}`) in the Conclusion section; verified IEEEtran packages (`amsmath`, `booktabs`, `cite`, `graphicx`, `hyperref`). |

---

## 2. Simulated Strict IEEE Access Reviewer Comments & Resolutions

1. **Reviewer Comment 1 (FGSM Non-Monotonicity):** *Why did single-step FGSM at $\epsilon=0.03$ yield higher accuracy than $\epsilon=0.01$ on ResNet-50 and EfficientNet-B0?*
   - **Resolution:** Audited `src/dataset.py` and `src/adversarial.py`, identified unshuffled class-0 (`buildings`) batch bias plus Kurakin et al.'s label-leaking effect from label smoothing ($\varepsilon_{\text{ls}}=0.05$), added class-stratified evaluation + predicted-label $\text{FGSM}_{\text{pred}}$ + dynamic 10-step PGD scaling, and explained the diagnostic findings explicitly in Section V.C.
2. **Reviewer Comment 2 (Statistical Rigor):** *Are the accuracy differences between Swin-Tiny, ResNet-50, and CLIP statistically significant?*
   - **Resolution:** Added 3-seed evaluations (`Mean ± Std`) on Intel Natural Scenes and paired McNemar tests ($\chi^2_{\text{cc}}$ and exact binomial $p$-values) in Table IV and Section V.D (`SUN397-100`).
3. **Reviewer Comment 3 (Scope & Compute Protocol Transparency):** *Clarify the exact sample counts used for SUN397-100 XAI and robustness.*
   - **Resolution:** Explicitly documented in Section IV.A, Table VI caption, and Section VI.C (`Limitations`): 5,000 held-out test images (50/class across 100 classes) for classification, 200 stratified test images (10 perturbation steps) for Insertion/Deletion AUC, and 500 stratified test images for FGSM/PGD.
4. **Reviewer Comment 4 (Compilation Note):**
   - **Status:** `pdflatex` is not installed on the local Windows environment; `paper_overleaf.zip` has been packaged with all relative paths (`figures/`, `tables/`, `references.bib`) ready for one-click Overleaf compilation.
