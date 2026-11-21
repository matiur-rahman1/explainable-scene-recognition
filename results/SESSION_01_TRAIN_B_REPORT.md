# Session S1-train-b Report: Swin-Tiny, CLIP-Linear & CLIP-ZeroShot Full-Dataset Intel Training (`SESSION_01_TRAIN_B_REPORT.md`)

**Session ID:** `S1-train-b` (completing unified `S1-train` across all 6 architectures)  
**Date:** 2026-11-21  
**Execution Environment:** Kaggle (`Tesla T4` x2, `14.56 GB` VRAM, PyTorch `2.11.0+cu130`, timm `1.0.30`, open_clip `4.2.2`, git commit `9fa0055`)  
**Protocol:** Full Intel Natural Scenes training split (`11,932` train / `2,102` val / `3,000` test), `swin_tiny` (6 epochs, `lr=2e-4`), `clip_linear` (5 epochs, `lr=1e-3`, frozen ViT-B/16 + linear head initialized from 5-prompt zero-shot text prototypes), `clip_zeroshot` (0 epochs, 5-prompt ensemble baseline), AMP training, strict **FP32** evaluation (`eval_fp32=True`).

---

## 1. Per-Unit Execution Summary for `S1-train-b` (All 7 Executed Units + 2 Deterministic Zero-Shot Mirrors)

| Unit ID | Model | Seed | Best Epoch | Best Val Acc (%) | Test Acc (%) | Macro-F1 (%) | Weighted-F1 (%) | Test Loss | FP32 Latency (ms) | FP16 Ckpt (MB) | Wall Time (s / min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `s1_train_intel_swin_tiny_seed0` | `swin_tiny` | `0` | 5/6 | 95.29% | **94.17%** | 94.26% | 94.14% | 0.3841 | 12.46 | 52.55 | 932.5s (15.54m) |
| `s1_train_intel_swin_tiny_seed1` | `swin_tiny` | `1` | 6/6 | 95.67% | **94.40%** | 94.50% | 94.38% | 0.3802 | 11.89 | 52.55 | 556.5s (9.27m) |
| `s1_train_intel_swin_tiny_seed2` | `swin_tiny` | `2` | 6/6 | 95.43% | **94.63%** | 94.75% | 94.63% | 0.3794 | 12.01 | 52.55 | 556.2s (9.27m) |
| `s1_train_intel_clip_linear_seed0` | `clip_linear` | `0` | 5/5 | 94.86% | **93.97%** | 94.11% | 93.94% | 0.4185 | 16.61 | 164.46 | 363.3s (6.05m) |
| `s1_train_intel_clip_linear_seed1` | `clip_linear` | `1` | 4/5 | 94.77% | **93.93%** | 94.07% | 93.91% | 0.4187 | 16.69 | 164.46 | 363.0s (6.05m) |
| `s1_train_intel_clip_linear_seed2` | `clip_linear` | `2` | 4/5 | 94.48% | **94.13%** | 94.26% | 94.11% | 0.4192 | 16.74 | 164.46 | 361.6s (6.03m) |
| `s1_train_intel_clip_zeroshot_seed0` | `clip_zeroshot` | `0` | 1/0 | 93.29% | **92.70%** | 92.80% | 92.65% | 0.2240 | 16.68 | 164.46 | 66.7s (1.11m) |
| `s1_train_intel_clip_zeroshot_seed1` | `clip_zeroshot` | `1` | 1/0 | 93.29% | **92.70%** | 92.80% | 92.65% | 0.2240 | 16.68 | 0.00 (mirror) | 0.0s (0.00m) |
| `s1_train_intel_clip_zeroshot_seed2` | `clip_zeroshot` | `2` | 1/0 | 93.29% | **92.70%** | 92.80% | 92.65% | 0.2240 | 16.68 | 0.00 (mirror) | 0.0s (0.00m) |

- **Total Wall Time Across `S1-train-b`:** `3199.7 seconds` (**`53.33 minutes`** = `0.89 hours`).
- **Total FP16 Checkpoints Size in `S1-train-b`:** **`815.49 MB`** across 7 `.pth` files.

---

## 2. Complete 6-Architecture Unified 3-Seed Benchmark on Intel Natural Scenes (`s1_train_full_intel_6models_summary.json`)

Combining `S1-train-a` and `S1-train-b` (all 6 architectures trained on the exact same `11,932` training images per seed and evaluated in strict FP32 on all `3,000` test images):

| Rank | Architecture | Paradigm | Params (Total / Trainable) | FP32 Latency (ms) | Test Accuracy (% Mean ± Std) | Macro-F1 (% Mean ± Std) | Weighted-F1 (% Mean ± Std) | Per-Seed Test Accuracies (Seed 0, 1, 2) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`swin_tiny`** | Hierarchical Shifted-Window ViT | 27.52M / 27.52M | 12.12 | **94.40 ± 0.19%** | **94.50 ± 0.20%** | 94.38 ± 0.20% | 94.17%, 94.40%, 94.63% |
| **2** | **`convnext_tiny`** | Modernized 7x7 Depthwise ConvNet | 27.82M / 27.82M | 6.30 | **94.27 ± 0.09%** | **94.40 ± 0.12%** | 94.26 ± 0.10% | 94.20%, 94.20%, 94.40% |
| **3** | **`resnet50`** | Classical Residual Bottleneck CNN | 23.52M / 23.52M | 6.02 | **94.19 ± 0.09%** | **94.31 ± 0.09%** | 94.16 ± 0.09% | 94.23%, 94.07%, 94.27% |
| **4** | **`clip_linear`** | Contrastive VLM (Linear Probe) | 86.20M / 0.00M | 16.68 | **94.01 ± 0.09%** | **94.15 ± 0.08%** | 93.99 ± 0.09% | 93.97%, 93.93%, 94.13% |
| **5** | **`efficientnet_b0`** | Compound-Scaled Mobile CNN | 4.02M / 4.02M | 8.40 | **93.69 ± 0.28%** | **93.81 ± 0.27%** | 93.66 ± 0.29% | 93.97%, 93.30%, 93.80% |
| **6** | **`clip_zeroshot`** | Contrastive VLM (Zero-Shot 5-Prompt) | 86.19M / 0.00M | 16.68 | **92.70 ± 0.00%** | **92.80 ± 0.00%** | 92.65 ± 0.00% | 92.70%, 92.70%, 92.70% |

---

## 3. All 15 Pairwise McNemar Tests With Holm–Bonferroni Correction (Seed 0 Canonical & 3-Seed Pooled)

Every single pair below is computed from the exact same `3,000`-image test set predictions (`n11 + n10 + n01 + n00 = 3,000` per seed; `9,000` pooled across 3 seeds), completely resolving the Stage 1 vs. Stage 3 mismatch found in `S0`:

| Model Pair ($A$ vs. $B$) | Seed 0 $n_{10} / n_{01}$ | Seed 0 $\chi^2_{\text{cc}}$ | Seed 0 Exact $p$ | Seed 0 Holm $p$ | Sig. (Holm $.05$) | 3-Seed Pooled $n_{10} / n_{01}$ ($N=9,000$) | Pooled Exact $p$ | Pooled Holm $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `swin_tiny_vs_clip_zeroshot` | 127 / 83 | 8.8048 | 0.002915 | 0.034980 | **Yes** | 376 / 223 | 0.000000 | 0.000000 |
| `convnext_tiny_vs_clip_zeroshot` | 123 / 78 | 9.6318 | 0.001841 | 0.023933 | **Yes** | 385 / 244 | 0.000000 | 0.000000 |
| `resnet50_vs_clip_zeroshot` | 117 / 71 | 10.7713 | 0.000979 | 0.013706 | **Yes** | 354 / 220 | 0.000000 | 0.000000 |
| `clip_linear_vs_clip_zeroshot` | 52 / 14 | 20.7424 | 0.000003 | 0.000045 | **Yes** | 159 / 41 | 0.000000 | 0.000000 |
| `efficientnet_b0_vs_clip_zeroshot` | 122 / 84 | 6.6456 | 0.009770 | 0.107470 | No | 361 / 272 | 0.000460 | 0.005060 |
| `swin_tiny_vs_efficientnet_b0` | 66 / 60 | 0.1984 | 0.656178 | 1.000000 | No | 227 / 163 | 0.001391 | 0.013910 |
| `convnext_tiny_vs_efficientnet_b0` | 61 / 54 | 0.3130 | 0.576012 | 1.000000 | No | 220 / 168 | 0.009533 | 0.085797 |
| `resnet50_vs_efficientnet_b0` | 67 / 59 | 0.3889 | 0.533053 | 1.000000 | No | 219 / 174 | 0.026326 | 0.210608 |
| `swin_tiny_vs_clip_linear` | 86 / 80 | 0.1506 | 0.698084 | 1.000000 | No | 250 / 215 | 0.114765 | 0.803355 |
| `clip_linear_vs_efficientnet_b0` | 81 / 81 | 0.0000 | 1.000000 | 1.000000 | No | 265 / 236 | 0.210911 | 1.000000 |
| `convnext_tiny_vs_clip_linear` | 82 / 75 | 0.2293 | 0.632185 | 1.000000 | No | 258 / 235 | 0.321770 | 1.000000 |
| `swin_tiny_vs_resnet50` | 60 / 62 | 0.0082 | 0.927911 | 1.000000 | No | 187 / 168 | 0.339416 | 1.000000 |
| `resnet50_vs_clip_linear` | 75 / 67 | 0.3451 | 0.557071 | 1.000000 | No | 226 / 210 | 0.472571 | 1.000000 |
| `swin_tiny_vs_convnext_tiny` | 48 / 49 | 0.0000 | 1.000000 | 1.000000 | No | 153 / 141 | 0.521247 | 1.000000 |
| `convnext_tiny_vs_resnet50` | 58 / 59 | 0.0000 | 1.000000 | 1.000000 | No | 188 / 181 | 0.754825 | 1.000000 |

---

## 4. Key Surprises & Insights from `S1-train-b` (and Unified 6-Model Comparison)

1. **True Clean Accuracy Ranking on Intel Natural Scenes (`11,932` Train / `3,000` Test, 3 Seeds):**
   - **1st:** `swin_tiny` (**`94.40 ± 0.19%`**, Macro-F1 `94.50 ± 0.20%`)
   - **2nd:** `convnext_tiny` (**`94.27 ± 0.09%`**, Macro-F1 `94.40 ± 0.12%`)
   - **3rd:** `resnet50` (**`94.19 ± 0.09%`**, Macro-F1 `94.31 ± 0.09%`)
   - **4th:** `clip_linear` (**`94.01 ± 0.09%`**, Macro-F1 `94.15 ± 0.08%`)
   - **5th:** `efficientnet_b0` (**`93.69 ± 0.28%`**, Macro-F1 `93.81 ± 0.27%`)
   - **6th:** `clip_zeroshot` (**`92.70 ± 0.00%`**, Macro-F1 `92.80 ± 0.00%`)
   *(Note: Previously in the Abstract, `ConvNeXt-Tiny` at `93.82%` was erroneously listed ahead of `CLIP-Linear` at `93.97%` because `ConvNeXt-Tiny` had only been trained on a 4,800-image subset. Now, under identical full-dataset training, `ConvNeXt-Tiny` genuinely ranks 2nd at `94.27 ± 0.09%`, ahead of both `ResNet-50` and `CLIP-Linear`!)*
2. **Consistent McNemar Counts (`swin_tiny_vs_convnext_tiny`):** On Seed 0 (`94.17%` = `2825/3000` for `swin_tiny` vs. `94.20%` = `2826/3000` for `convnext_tiny`), `swin_tiny_vs_convnext_tiny` has `n11=2777, n10=48, n01=49, n00=126` ($p=1.000$), and across all 3 seeds pooled ($N=9,000$), `n10/n01 = 153 / 141` ($p=0.5212$, Holm $p=1.000$). Indeed, the top four models (`swin_tiny`, `convnext_tiny`, `resnet50`, `clip_linear`) have no statistically significant pairwise difference after Holm–Bonferroni correction on single seeds or pooled, whereas all five trained models significantly outperform `clip_zeroshot` in the 3-seed pooled test ($p \le 0.0051$ Holm), and `swin_tiny` significantly outperforms `efficientnet_b0` ($p = 0.0139$ Holm).
3. **Kaggle Nested Mount Discovery:** In `S1-train-b`, Kaggle mounted the attached `S1-train-a` notebook output deeper under `/kaggle/input/...` than 1 directory level, so `S1-train-b` did not see the `S1-train-a` files inside Kaggle's container (though we had all 9 `S1-train-a` JSONs locally and merged them here into `s1_train_full_intel_6models_summary.json`). To prevent this in `S1a` and all future sessions, we upgraded checkpoint discovery to use **recursive `Path('/kaggle/input').rglob('best_*_intel_seed*.pth')`** so checkpoints are found at any directory depth.
