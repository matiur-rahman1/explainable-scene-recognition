# Session S0 Report: Local Codebase, Data & Manuscript Audit (`SESSION_00_REPORT.md`)

**Session Type:** `S0 [LOCAL, CPU only, no GPU]`  
**Date:** 2026-11-21  
**Outputs Produced:**
1. `results/REPRO_MAP.md` (complete provenance map of Tables 1–7 and Figures 1–7)
2. `results/delta_auc_all_methods.csv` (96 rows: recomputed $\Delta\text{AUC} = \text{AUC}_{\text{ins}} - \text{AUC}_{\text{del}}$ across all 4 CAM methods, all 6 architectures, Intel and SUN397-100, clean and defended)
3. `results/SESSION_00_REPORT.md` (this audit report and GPU session budget plan)

---

## 1. Recomputed $\Delta\text{AUC}$ Across All 4 CAM Methods (`results/delta_auc_all_methods.csv`)

In the current manuscript draft, Table 3 (`tab:intel_multiseed`) and Table 7 (`tab:sun397_results`) only reported $\Delta\text{AUC} = \text{AUC}_{\text{ins}} - \text{AUC}_{\text{del}}$ for `VG-LayerCAM`, omitting $\Delta\text{AUC}$ for `Grad-CAM`, `Grad-CAM++`, and `LayerCAM`. Recomputing $\Delta\text{AUC}$ for all four methods from `results/multi_seed_intel.json`, `results/stage3_intel_convnext_vglayercam.json`, `results/multi_seed_sun397_100.json`, and `results/stage3_sun397_convnext_vglayercam_sac.json` yields:

### 1.1 Intel Natural Scenes (Clean 3-Seed Mean $\pm$ Std, $S=20$ steps, $N=60$ stratified images)

| Architecture | Grad-CAM $\Delta\text{AUC}$ (Ins / Del) | Grad-CAM++ $\Delta\text{AUC}$ (Ins / Del) | LayerCAM $\Delta\text{AUC}$ (Ins / Del) | VG-LayerCAM $\Delta\text{AUC}$ (Ins / Del) | True Best $\Delta\text{AUC}$ Method |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`resnet50`** | **`+0.1945 ± 0.0252`** (`0.8015 / 0.6071`) | `+0.1774 ± 0.0235` (`0.7967 / 0.6193`) | `+0.1693 ± 0.0256` (`0.7935 / 0.6242`) | `+0.1891 ± 0.0287`* (`0.7523 / 0.5632`) | **Grad-CAM** (VG-LayerCAM best Del↓) |
| **`efficientnet_b0`** | `+0.1393 ± 0.0250` (`0.7787 / 0.6395`) | `-0.0724 ± 0.0287` (`0.6760 / 0.7484`) | `+0.0021 ± 0.0285` (`0.7115 / 0.7093`) | **`+0.1818 ± 0.0358`*** (`0.7028 / 0.5210`) | **VG-LayerCAM*** (via Del↓ `0.5210`) |
| **`convnext_tiny`** | **`+0.2825 ± 0.0268`** (`0.8657 / 0.5831`) | `+0.0439 ± 0.0323` (`0.7589 / 0.7150`) | `+0.1025 ± 0.0671` (`0.8110 / 0.7084`) | `+0.2819 ± 0.0226` (`0.8561 / 0.5742`) | **Grad-CAM** by `+0.0006` (VG-LayerCAM beats LayerCAM by `+0.1794` & best Del↓) |
| **`swin_tiny`** | `-0.0630 ± 0.0342` (`0.6854 / 0.7484`) | `+0.1247 ± 0.0229` (`0.7810 / 0.6563`) | `+0.1158 ± 0.0247` (`0.7964 / 0.6807`) | **`+0.1898 ± 0.0139`*** (`0.8175 / 0.6278`) | **VG-LayerCAM** (best Ins↑, Del↓, $\Delta\text{AUC}$↑) |
| **`clip_zeroshot`** | **`+0.1147 ± 0.0000`** (`0.8312 / 0.7165`) | `+0.0458 ± 0.0000` (`0.7937 / 0.7479`) | `-0.0775 ± 0.0000` (`0.7408 / 0.8183`) | `-0.0495 ± 0.0000` (`0.7443 / 0.7939`) | **Grad-CAM** (LayerCAM & VG-LayerCAM negative!) |
| **`clip_linear`** | **`+0.1043 ± 0.0010`** (`0.7997 / 0.6954`) | `+0.0201 ± 0.0315` (`0.7539 / 0.7338`) | `-0.0874 ± 0.0044` (`0.7081 / 0.7955`) | `-0.0674 ± 0.0128`* (`0.7155 / 0.7829`) | **Grad-CAM** (LayerCAM & VG-LayerCAM negative!) |

*\*Warning (Protocol Discrepancy Found):* For `resnet50`, `efficientnet_b0`, `swin_tiny`, and `clip_linear`, the `Grad-CAM`, `Grad-CAM++`, and `LayerCAM` columns were evaluated on Stage 1 checkpoints (trained for **6 epochs on all 11,932 training images**), whereas `VG-LayerCAM` was evaluated in Stage 3 (`scripts/run_stage3_neurocomputing.py`, lines 221–235), which retrained those models for **3 epochs on a 4,800-image subset** because the Stage 1 checkpoints were not persisted across Kaggle sessions! Only `convnext_tiny` and `clip_zeroshot` evaluated all 4 methods on the exact same model weights.

### 1.2 SUN397-100 (Clean Seed 0, $S=10$ steps, $N=200$ stratified images)

| Architecture | Grad-CAM $\Delta\text{AUC}$ (Ins / Del) | Grad-CAM++ $\Delta\text{AUC}$ (Ins / Del) | LayerCAM $\Delta\text{AUC}$ (Ins / Del) | VG-LayerCAM $\Delta\text{AUC}$ (Ins / Del) | True Best $\Delta\text{AUC}$ Method |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`resnet50`** | **`+0.2325`** (`0.3618 / 0.1293`) | `+0.2167` (`0.3507 / 0.1340`) | `+0.2130` (`0.3489 / 0.1359`) | `+0.1641`* (`0.2748 / 0.1107`) | **Grad-CAM** (VG-LayerCAM best Del↓) |
| **`efficientnet_b0`** | **`+0.2600`** (`0.4436 / 0.1836`) | `+0.2176` (`0.4172 / 0.1996`) | `+0.2167` (`0.4134 / 0.1967`) | `+0.2105`* (`0.4008 / 0.1903`) | **Grad-CAM** |
| **`convnext_tiny`** | **`+0.3671`** (`0.5565 / 0.1895`) | `+0.1960` (`0.4633 / 0.2673`) | `+0.1963` (`0.4905 / 0.2942`) | `+0.3584` (`0.5430 / 0.1846`) | **Grad-CAM** by `+0.0087` (VG-LayerCAM beats LayerCAM by `+0.1621` & best Del↓) |
| **`swin_tiny`** | `+0.1538` (`0.5518 / 0.3980`) | `+0.0352` (`0.4861 / 0.4509`) | `+0.1865` (`0.5724 / 0.3859`) | **`+0.2547`*** (`0.5636 / 0.3089`) | **VG-LayerCAM** (by `+0.0682` over LayerCAM) |
| **`clip_zeroshot`** | **`+0.1434`** (`0.4520 / 0.3086`) | `+0.0153` (`0.3792 / 0.3639`) | `-0.0642` (`0.3368 / 0.4010`) | `-0.0520` (`0.3403 / 0.3923`) | **Grad-CAM** |
| **`clip_linear`** | **`+0.1564`** (`0.4751 / 0.3187`) | `+0.0268` (`0.3970 / 0.3702`) | `-0.0689` (`0.3504 / 0.4193`) | `-0.0532` (`0.3554 / 0.4086`) | **Grad-CAM** |

*\*Same protocol note on SUN397-100:* `resnet50`, `efficientnet_b0`, and `swin_tiny` were trained for 5 epochs in Stage 2 (`multi_seed_sun397_100.json`), whereas Stage 3 (`run_stage3_neurocomputing.py`, line 466) retrained them for 4 epochs when evaluating `VG-LayerCAM`.

---

## 2. Root Cause of Table 4 McNemar Discrepancy (`ConvNeXt vs. Swin: 49 / 53`)

**Question:** Why does Table 4 show `ConvNeXt-Tiny vs. Swin-Tiny: n10/n01 = 49 / 53` (a difference of only $4 / 3000 = 0.133\%$ in test accuracy), whereas Table 3 reports `93.82%` vs. `94.40%` (a $0.58\%$ gap, or `93.60%` vs. `94.63%` = $1.03\%$ gap on Seed 0)?

**Exact Code & Data Diagnosis:**
1. In **Stage 1** (`main.py` $\to$ `results/multi_seed_intel.json` and `results/mcnemar_intel.json`), the initial 5 models were trained on all **11,932 Intel training images for 6 epochs** (`5` epochs for `clip_linear`). On Seed 0:
   - `swin_tiny` achieved `94.63%` (`2,839 / 3,000` correct).
   - `resnet50` achieved `94.20%` (`2,826 / 3,000` correct).
   - `clip_linear` achieved `94.00%` (`2,820 / 3,000` correct).
   - `efficientnet_b0` achieved `93.73%` (`2,812 / 3,000` correct).
   - `clip_zeroshot` achieved `92.70%` (`2,781 / 3,000` correct).
   The first 10 rows of Table 4 were taken directly from `results/mcnemar_intel.json` (e.g., `Swin-Tiny vs. ResNet-50: n11=2771, n10=68, n01=55`, where $2771+68 = 2839$ and $2771+55 = 2826$).
2. In **Stage 3** (`scripts/run_stage3_neurocomputing.py` $\to$ `results/stage3_intel_convnext_vglayercam.json`), the script ran in a new Kaggle session where the Stage 1 `.pth` checkpoints were not available. As shown in `run_stage3_neurocomputing.py` lines 124–146 and 221–255:
   - `convnext_tiny` was trained on a **4,800-image stratified subset for 5 epochs**, achieving `93.60%` (`2,808 / 3,000` correct) on Seed 0 (`93.82 ± 0.25%` across 3 seeds).
   - To compute `mcnemar_6models_intel` in Stage 3 (line 275), `run_stage3_neurocomputing.py` retrained `swin_tiny` (and `resnet50`, `efficientnet_b0`, `clip_linear`) on the **4,800-image subset for only 3 epochs** (`ep = 3`, line 221)!
   - In that 3-epoch 4,800-image run, `swin_tiny` achieved `2,759 + 53 = 2,812 / 3,000 = 93.73%` (instead of `2,839 / 3,000 = 94.63%`), while `convnext_tiny` achieved `2,759 + 49 = 2,808 / 3,000 = 93.60%`.
   - Table 4 in `paper/main.tex` concatenated the 10 rows from `results/mcnemar_intel.json` (Stage 1: 6-epoch 11,932-image models) with 1 row (`ConvNeXt-Tiny vs. Swin-Tiny: 49 / 53`) from `results/stage3_intel_convnext_vglayercam.json` (Stage 3: 5-epoch ConvNeXt vs. 3-epoch Swin on 4,800 images)!

**Fix:** In `S1-train`, all trainable Intel models (`resnet50`, `efficientnet_b0`, `convnext_tiny`, `swin_tiny`, `clip_linear`) will be trained under the **exact same protocol** (full `11,932` training images, `6` epochs for backbones, `5` epochs for `clip_linear`, seeds `{0, 1, 2}`), and each unit will save its full 3,000-image test prediction vector `y_pred` so that all $\binom{6}{2} = 15$ pairwise McNemar tests (with Holm–Bonferroni correction) are computed from the exact same checkpoints that produce Table 3.

---

## 3. Why Table 2 (`tab:intel_prelim`) Disagrees with Tables 3 and 5 — and Decision

**Question:** Why does Table 2 report `Swin-Tiny Grad-CAM++ Del = 0.4899` (vs. `0.6563 ± 0.0114` in Table 3) and `ResNet-50 PGD(0.03) = 19.17%` (vs. `2.44 ± 0.55%` in Table 5)?

**Diagnosis:**
Table 2 (`tab:intel_prelim`) is a leftover table from an early preliminary 3-model run (`seed=42`) executed before two critical bug fixes were made to the codebase:
1. **Unstratified Test Slicing:** Before `build_stratified_eval_loader` was added to `src/dataset.py`, `evaluate_xai_faithfulness` and `evaluate_adversarial_robustness` iterated directly over `test_loader` (`shuffle=False`) and stopped after 60 images (XAI) or 15 batches = 480 images (adversarial). Because `ImageFolder` orders classes alphabetically (`buildings` = class 0 has 437 test images), 100% of the 60 XAI images and 91% of the 480 adversarial images belonged to a single class (`buildings`).
2. **Unscaled PGD Perturbation in Normalized Space:** In the preliminary run, PGD applied $\epsilon=0.03$ directly to ImageNet-normalized tensors $\tilde{x} = (x - \mu)/\sigma$ without dividing by $\sigma = (0.229, 0.224, 0.225)$. Thus, the actual pixel-space perturbation budget was only $\epsilon_{\text{pixel}} = 0.03 \times 0.224 \approx 0.0067$ (worse than $\epsilon=0.01$), which inflated ResNet-50's apparent PGD survival to `19.17%`. Once `_norm_bounds` (`eps_scaled = epsilon / std`) was implemented in `src/adversarial.py`, 10-step PGD($\epsilon=0.03$) accuracy dropped to `2.44 ± 0.55%` (Table 5).

**Decision:** **Delete Table 2 (`tab:intel_prelim`) completely** in `S10`, and remove all references to it in Section 5.1, Section 5.2, and Section 5.3.

---

## 4. Additional Surprises & Critical Audit Findings to Address in S1–S10

1. **Zero Local Checkpoints Exist (`checkpoints/` is missing):**
   - Because all prior runs executed on Kaggle and only `results/*.json` and `outputs/*.png` were downloaded, **no `.pth` model checkpoints exist locally**.
   - Therefore, we must run **`S1-train`** sessions first to train and save fp16 `state_dict` checkpoints (plus full `y_true`/`y_pred` and per-epoch `history`) for all Intel models across seeds `{0, 1, 2}` on the full `11,932`-image training set.
2. **SAC-PGD-AT vs. Standard PGD-AT ($\beta=0$) Honest Data Check (`results/stage3_sac_ablation_intel.json` & `stage3_sun397_convnext_vglayercam_sac.json`):**
   - Looking at `results/stage3_sac_ablation_intel.json` (Seed 0, 3 epochs on 3,600 images):
     - **On `convnext_tiny` (Intel):** Clean accuracy is `90.70%` ($\beta=0.0$), `90.70%` ($\beta=0.05$), `90.77%` ($\beta=0.1$), `90.77%` ($\beta=0.2$). PGD($0.03$) is **`62.33%` for ALL four $\beta \in \{0.0, 0.05, 0.1, 0.2\}$**! PGD($0.05$) ticks up slightly from `35.33%` ($\beta=0.0$) to `36.50%` ($\beta=0.2$). Clean-vs-PGD CAM cosine similarity is already `0.9865` at $\beta=0.0$ and `0.9872` at $\beta=0.2$ (a difference of `+0.0007`, well within noise `std=0.018`).
     - **On `swin_tiny` (Intel):** Clean accuracy is `93.50%` ($\beta=0.0$) vs. `93.47%` ($\beta=0.05, 0.2$) and `93.37%` ($\beta=0.1$). PGD($0.03$) is `36.33%` ($\beta=0.0$), drops to `35.33%` ($\beta=0.05$), and is `36.83%` ($\beta=0.1$) and `37.50%` ($\beta=0.2$). Meanwhile, clean-vs-PGD CAM cosine similarity is **NOT monotonic** in $\beta$: it goes `0.7667` ($\beta=0.0$) $\to$ `0.7945` ($\beta=0.05$) $\to$ `0.7635` ($\beta=0.1$, *lower* than $\beta=0.0$!) $\to$ `0.7706` ($\beta=0.2$). And `VG-LayerCAM` Insertion AUC is `0.8367` at $\beta=0.0$ vs. `0.8368` at $\beta=0.1$ (a difference of `+0.0001`).
     - **On `resnet50` (Intel):** Clean accuracy (`92.07%`) and PGD($0.03$) (`11.50%`) are identical at $\beta=0.0$ and $\beta=0.1$, while CAM cosine similarity rises from `0.8587` ($\beta=0.0$) to `0.8914` ($\beta=0.1$).
     - **On `SUN397-100` (`stage3_sun397_convnext_vglayercam_sac.json`):** For `convnext_tiny`, PGD($0.01$) is `48.20%` and PGD($0.03$) is `10.80%` at **both** $\beta=0.0$ and $\beta=0.1$ (the massive jump from `0.00%` to `10.80%` and the clean accuracy drop from `79.92%` to `66.86%`/`67.00%` come from **standard PGD-AT ($\beta=0$)**, not from the SAC penalty!). For `swin_tiny`, PGD($0.03$) is `0.00%` at both $\beta=0.0$ and $\beta=0.1$.
   - **Implication (Hard Rule 2):** In `S4` (3 seeds) and `S9`/`S10`, we must test whether $\beta > 0$ provides any statistically significant improvement over $\beta = 0$, report the exact 3-seed results honestly, and reframe SAC-PGD-AT as an attribution-stability regularization analysis rather than claiming SAC itself drives the PGD accuracy jump or exhibits strictly monotonic test-set CAM similarity.
3. **Mathematical Identity for SAC Penalty:**
   - Because $\bar{E}(x)$ and $\bar{E}(x_{\text{PGD}})$ are L2-normalized to unit Frobenius norm ($\|\bar{E}(x)\|_F = \|\bar{E}(x_{\text{PGD}})\|_F = 1$),
     $$\|\bar{E}(x) - \bar{E}(x_{\text{PGD}})\|_F^2 = \|\bar{E}(x)\|_F^2 + \|\bar{E}(x_{\text{PGD}})\|_F^2 - 2\langle \bar{E}(x), \bar{E}(x_{\text{PGD}})\rangle = 2\bigl(1 - \cos(\bar{E}(x), \bar{E}(x_{\text{PGD}}))\bigr).$$
   - Also, `VG-LayerCAM` channel weights $w_k^c = \frac{\sum_{i,j} (G_{k,i,j}^c)^2 \text{ReLU}(A_{k,i,j})}{\sum_{i,j} (G_{k,i,j}^c)^2 + \delta}$ are a **second-order gradient-energy-weighted activation average** (not a statistical variance). In `S10`, we will rename/clarify it as **Gradient-Energy-Gated LayerCAM (`GE-LayerCAM` / `VG-LayerCAM`)** and state this explicitly.

---

## 5. Complete GPU Session Plan & Estimated Wall-Clock Hours (Target $\le 2.5\text{h}$/session, Hard Stop $3.0\text{h}$)

Because no trained checkpoints exist locally, we insert **`S1-train-a`** and **`S1-train-b`** before `S1a`/`S1b` so all 5 trainable Intel architectures (`resnet50`, `efficientnet_b0`, `convnext_tiny`, `swin_tiny`, `clip_linear`) are trained for the full 6 epochs (5 for `clip_linear`) on all `11,932` Intel training images across seeds `{0, 1, 2}`, saving compact FP16 `state_dict` checkpoints (`~15–110 MB` each, total `~1.1 GB` across all 15 checkpoints) and full `3,000`-image test predictions.

| Session ID | Scope & Units (`1 unit = 1 (dataset, model, seed, config)`) | Est. Wall Time on Kaggle T4 (hours) | Cumulative GPU Hours | Outputs Saved & Persisted |
| :--- | :--- | :---: | :---: | :--- |
| **`S0`** | **[LOCAL CPU — DONE]** Audit, `REPRO_MAP.md`, `delta_auc_all_methods.csv`, `SESSION_00_REPORT.md` | `0.0 h` | `0.0 h` | `results/REPRO_MAP.md`, `results/delta_auc_all_methods.csv`, `results/SESSION_00_REPORT.md` |
| **`S1-train-a`** | **[GPU]** Intel full-train (`11,932` imgs, 6 epochs, AMP train, FP32 eval): `resnet50`, `efficientnet_b0`, `convnext_tiny` $\times$ seeds `{0, 1, 2}` (9 units) + clean test eval (`3,000` imgs, `y_pred` saved) | `1.6 h` | `1.6 h` | `results/s1_train_<model>_seed<s>.json`, `checkpoints/best_<model>_intel_seed<s>.pth` (fp16) |
| **`S1-train-b`** | **[GPU]** Intel full-train (`11,932` imgs): `swin_tiny` (6 ep), `clip_linear` (5 ep) $\times$ seeds `{0, 1, 2}` + `clip_zeroshot` (seed 0) (7 units) + clean test eval (`3,000` imgs, `y_pred` saved) | `1.5 h` | `3.1 h` | `results/s1_train_<model>_seed<s>.json`, `checkpoints/best_<model>_intel_seed<s>.pth` (fp16) |
| **`S1a`** | **[GPU]** Intel full-test (`3,000` imgs, FP32, no AMP) `FGSM`, `FGSM_pred`, and `PGD-10` at $\epsilon \in \{0.01, 0.03, 0.05\}$: `resnet50`, `efficientnet_b0`, `convnext_tiny` $\times$ seeds `{0, 1, 2}` (9 units) | `1.8 h` | `4.9 h` | `results/s1_pgd_full_intel_<model>_seed<s>.json` |
| **`S1b`** | **[GPU]** Intel full-test (`3,000` imgs, FP32, no AMP) `FGSM`, `FGSM_pred`, and `PGD-10` at $\epsilon \in \{0.01, 0.03, 0.05\}$: `swin_tiny`, `clip_linear` $\times$ seeds `{0, 1, 2}` + `clip_zeroshot` (7 units) | `2.1 h` | `7.0 h` | `results/s1_pgd_full_intel_<model>_seed<s>.json` |
| **`S2a`** | **[GPU]** Intel AutoAttack (`standard`: `APGD-CE`, `APGD-T`, `FAB-T`, `Square`, FP32) at $\epsilon=0.03$ on `1,000` stratified test images: `resnet50`, `efficientnet_b0`, `convnext_tiny` (seed 0 first; seeds 1, 2 as time-guarded units) | `1.9 h` | `8.9 h` | `results/s2_autoattack_intel_<model>_seed<s>.json` |
| **`S2b`** | **[GPU]** Intel AutoAttack (`standard`, FP32) at $\epsilon=0.03$ on `1,000` stratified test images: `swin_tiny`, `clip_linear`, `clip_zeroshot` + gradient-masking sanity checks (Square black-box vs. APGD white-box) | `2.2 h` | `11.1 h` | `results/s2_autoattack_intel_<model>_seed<s>.json` |
| **`S3`** | **[GPU]** Label-smoothing ablation on Intel: retrain `resnet50` and `efficientnet_b0` with $\alpha_{\text{ls}} = 0.0$ (vs. $0.05$) $\times$ seeds `{0, 1, 2}` (6 units), evaluate FP32 true-label `FGSM` & `FGSM_pred` at $\epsilon \in \{0.01, 0.03, 0.05\}$ on full `3,000` test set | `1.5 h` | `12.6 h` | `results/s3_ls0_intel_<model>_seed<s>.json` |
| **`S4a`** | **[GPU]** SAC-PGD-AT on Intel (`resnet50`): $\beta \in \{0.0, 0.05, 0.1, 0.2\} \times$ seeds `{0, 1, 2}` (12 units; 3 epochs, warm-started from `S1-train`, full `3,000`-img clean & PGD `.01/.03/.05` eval + CAM Cosine, Spearman rank correlation, top-20% IoU under PGD-10) | `2.1 h` | `14.7 h` | `results/s4_sac_intel_resnet50_b<beta>_seed<s>.json`, best checkpoints |
| **`S4b`** | **[GPU]** SAC-PGD-AT on Intel (`convnext_tiny`): $\beta \in \{0.0, 0.05\} \times$ seeds `{0, 1, 2}` (6 units; full `3,000`-img PGD + CAM Cosine/Spearman/IoU) | `1.7 h` | `16.4 h` | `results/s4_sac_intel_convnext_tiny_b<beta>_seed<s>.json`, best checkpoints |
| **`S4c`** | **[GPU]** SAC-PGD-AT on Intel (`convnext_tiny`): $\beta \in \{0.1, 0.2\} \times$ seeds `{0, 1, 2}` (6 units; full `3,000`-img PGD + CAM Cosine/Spearman/IoU) | `1.7 h` | `18.1 h` | `results/s4_sac_intel_convnext_tiny_b<beta>_seed<s>.json`, best checkpoints |
| **`S4d`** | **[GPU]** SAC-PGD-AT on Intel (`swin_tiny`): $\beta \in \{0.0, 0.05\} \times$ seeds `{0, 1, 2}` (6 units; full `3,000`-img PGD + CAM Cosine/Spearman/IoU) | `1.9 h` | `20.0 h` | `results/s4_sac_intel_swin_tiny_b<beta>_seed<s>.json`, best checkpoints |
| **`S4e`** | **[GPU]** SAC-PGD-AT on Intel (`swin_tiny`): $\beta \in \{0.1, 0.2\} \times$ seeds `{0, 1, 2}` (6 units; full `3,000`-img PGD + CAM Cosine/Spearman/IoU) | `1.9 h` | `21.9 h` | `results/s4_sac_intel_swin_tiny_b<beta>_seed<s>.json`, best checkpoints |
| **`S5a`** | **[GPU]** `TRADES` ($\beta_{\text{TRADES}}=6.0$) and `MART` baselines on Intel: `resnet50` + `convnext_tiny` $\times$ seeds `{0, 1, 2}` (12 units; same epoch/step budget as `S4`, full-test PGD + CAM stability metrics) | `2.3 h` | `24.2 h` | `results/s5_<defense>_intel_<model>_seed<s>.json`, best checkpoints |
| **`S5b`** | **[GPU]** `TRADES` ($\beta_{\text{TRADES}}=6.0$) and `MART` baselines on Intel: `swin_tiny` $\times$ seeds `{0, 1, 2}` (6 units; full-test PGD + CAM stability metrics) | `1.9 h` | `26.1 h` | `results/s5_<defense>_intel_swin_tiny_seed<s>.json`, best checkpoints |
| **`S6a`** | **[GPU]** AutoAttack (`standard`, $\epsilon=0.03$, `1,000` stratified Intel images, FP32) on defended models (`PGD-AT` $\beta=0$, best `SAC-PGD-AT` $\beta$, `TRADES`, `MART`) for `resnet50` and `convnext_tiny` | `2.1 h` | `28.2 h` | `results/s6_autoattack_defended_intel_<model>_<def>_seed<s>.json` |
| **`S6b`** | **[GPU]** AutoAttack (`standard`, $\epsilon=0.03$, `1,000` stratified Intel images, FP32) on defended models (`PGD-AT` $\beta=0$, best `SAC-PGD-AT` $\beta$, `TRADES`, `MART`) for `swin_tiny` | `1.8 h` | `30.0 h` | `results/s6_autoattack_defended_intel_swin_tiny_<def>_seed<s>.json` |
| **`S7a`** | **[GPU]** Unified Intel XAI benchmark ($N=1,000$ stratified images, $S=20$ steps, FP32) across all 6 architectures (3 seeds from `S1-train` checkpoints): `Grad-CAM`, `Grad-CAM++`, `LayerCAM`, `VG-LayerCAM`, `HiResCAM`, `XGrad-CAM`, and `Random-Map` baseline | `2.2 h` | `32.2 h` | `results/s7a_xai_intel_<model>_seed<s>.json` + genuine 5+ col XAI grids |
| **`S7b`** | **[GPU]** Gradient-free & Transformer-specific XAI on Intel: `Score-CAM` ($N=500$ stratified images) across all 6 models + `Attention Rollout` / `Chefer Transformer Attribution` on `swin_tiny`, `clip_zeroshot`, `clip_linear` | `2.1 h` | `34.3 h` | `results/s7b_xai_transformer_scorecam_intel_<model>_seed<s>.json` |
| **`S7c`** | **[GPU]** Adebayo Sanity Checks (cascading model parameter randomization & label randomization; Spearman rank correlation & SSIM vs. original map) for `Grad-CAM`, `LayerCAM`, `VG-LayerCAM` on `resnet50`, `convnext_tiny`, `swin_tiny` | `1.2 h` | `35.5 h` | `results/s7c_sanity_checks_intel.json` + sanity progression figure |
| **`S7d`** | **[GPU]** Layer & Gate Ablation on Intel: `swin_tiny` across $\ge 2$ target layers (`layers[-1].blocks[-1].norm1` vs. earlier stage `layers[-2].blocks[-1].norm1`) + `CLIP ViT-B/16` signed vs. ReLU activation gate in `VG-LayerCAM` | `1.3 h` | `36.8 h` | `results/s7d_layer_gate_ablation_intel.json` |
| **`S8a`** | **[GPU]** `SUN397-100` clean training of all 6 architectures, **Seed 0** (pre-caches $224\times224$ tensors to disk first; `4,300` train, `700` val, `5,000` test; saves fp16 checkpoints, full `y_pred`, and FP32 PGD-10 on `1,000` stratified test images) | `1.8 h` | `38.6 h` | `results/s8_train_sun397_<model>_seed0.json`, `checkpoints/best_<model>_sun397_seed0.pth` |
| **`S8b`** | **[GPU]** `SUN397-100` clean training of all 5 trainable architectures, **Seeds 1 & 2** (10 units) + full `5,000`-image test eval & FP32 PGD-10 | `2.3 h` | `40.9 h` | `results/s8_train_sun397_<model>_seed<1,2>.json`, checkpoints |
| **`S8c`** | **[GPU]** `SUN397-100` XAI faithfulness ($N=500$ stratified images, $S=20$ steps matching Intel, FP32) across all 6 architectures $\times$ 3 seeds for all CAM methods (`Grad-CAM`, `Grad-CAM++`, `LayerCAM`, `VG-LayerCAM`, `HiResCAM`, `XGrad-CAM`, `Random`) | `2.2 h` | `43.1 h` | `results/s8c_xai_sun397_<model>_seed<s>.json` + genuine 5-col SUN397 XAI grids |
| **`S8d`** | **[GPU]** `SUN397-100` `PGD-AT` ($\beta=0.0$) vs. `SAC-PGD-AT` ($\beta=0.1$) for `convnext_tiny` and `swin_tiny` $\times$ **3 seeds** (`{0, 1, 2}`, 12 units) + full PGD eval & CAM stability metrics | `2.4 h` | `45.5 h` | `results/s8d_sac_sun397_<model>_b<beta>_seed<s>.json` |

*Quota Fitting Note:* Total GPU compute is `~45.5 GPU hours` across 24 short, self-contained sessions (`1.2h–2.4h` each, every single session $\le 2.5\text{h}$). This fits comfortably inside **1.5 weeks of a single Kaggle account's 30h/week quota** (or **1 week** by running `S1–S6` (`30.0h`) on Kaggle T4 and `S7–S8` (`15.5h`) across Colab Free T4 / a second week).

---

## 6. Readiness for Next Step

Session `S0` is complete. Waiting for your confirmation before writing the first GPU session script (`S1-train-a` / `S1a`).
