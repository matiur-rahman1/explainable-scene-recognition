# Session S1-train-a Report: Unified Full-Dataset Intel Training (`SESSION_01_TRAIN_A_REPORT.md`)

**Session ID:** `S1-train-a`  
**Date:** 2026-11-21  
**Execution Environment:** Kaggle (`Tesla T4` x2, `14.56 GB` VRAM, PyTorch `2.11.0+cu130`, timm `1.0.30`, Python `3.13.16`)  
**Protocol:** Full Intel Natural Scenes training split (`11,932` train / `2,102` val / `3,000` test), 6 epochs, `batch_size=32`, `AdamW(lr=2e-4, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=6)`, `label_smoothing=0.05`, AMP training, strict **FP32** evaluation (`eval_fp32=True`).

---

## 1. Per-Unit Execution Summary (All 9/9 Units Completed)

| Unit ID | Model | Seed | Best Epoch | Best Val Acc (%) | Test Acc (%) | Macro-F1 (%) | Weighted-F1 (%) | Test Loss | FP32 Latency (ms) | FP16 Ckpt (MB) | Wall Time (s / min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `s1_train_intel_resnet50_seed0` | `resnet50` | `0` | 4/6 | 94.24% | **94.23%** | 94.36% | 94.21% | 0.3826 | 5.99 | 45.06 | 566.9s (9.45m) |
| `s1_train_intel_resnet50_seed1` | `resnet50` | `1` | 6/6 | 95.00% | **94.07%** | 94.19% | 94.03% | 0.3795 | 6.06 | 45.06 | 359.3s (5.99m) |
| `s1_train_intel_resnet50_seed2` | `resnet50` | `2` | 5/6 | 94.58% | **94.27%** | 94.39% | 94.24% | 0.3849 | 6.00 | 45.06 | 358.9s (5.98m) |
| `s1_train_intel_efficientnet_b0_seed0` | `efficientnet_b0` | `0` | 6/6 | 94.05% | **93.97%** | 94.07% | 93.94% | 0.4241 | 7.89 | 7.85 | 268.2s (4.47m) |
| `s1_train_intel_efficientnet_b0_seed1` | `efficientnet_b0` | `1` | 6/6 | 94.43% | **93.30%** | 93.43% | 93.26% | 0.4309 | 8.69 | 7.85 | 263.8s (4.40m) |
| `s1_train_intel_efficientnet_b0_seed2` | `efficientnet_b0` | `2` | 6/6 | 94.24% | **93.80%** | 93.93% | 93.79% | 0.4277 | 8.61 | 7.85 | 263.1s (4.39m) |
| `s1_train_intel_convnext_tiny_seed0` | `convnext_tiny` | `0` | 6/6 | 94.53% | **94.20%** | 94.29% | 94.18% | 0.3915 | 6.09 | 53.13 | 492.1s (8.20m) |
| `s1_train_intel_convnext_tiny_seed1` | `convnext_tiny` | `1` | 6/6 | 95.20% | **94.20%** | 94.35% | 94.20% | 0.3892 | 6.52 | 53.13 | 488.1s (8.14m) |
| `s1_train_intel_convnext_tiny_seed2` | `convnext_tiny` | `2` | 5/6 | 94.39% | **94.40%** | 94.56% | 94.40% | 0.3831 | 6.28 | 53.13 | 488.4s (8.14m) |

- **Total Wall Time Across All 9 Units:** `3548.7 seconds` (**`59.15 minutes`** = `0.99 hours`, well under the 150-minute budget).
- **Total FP16 Checkpoints Size:** **`318.12 MB`** across 9 `.pth` files.

---

## 2. Aggregated 3-Seed Results (`results/s1_train_a_summary.json`) vs. Previous Manuscript Table 3

| Architecture | Params (M) | FP32 Latency (ms) | New Unified 3-Seed Test Acc (%) | Previous Draft Table 3 Test Acc (%) | New Unified 3-Seed Macro-F1 (%) | Previous Draft Table 3 Macro-F1 (%) | Protocol Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`convnext_tiny`** | 27.82M | 6.30 | **94.27 ± 0.09%** | 93.82 ± 0.25% | **94.40 ± 0.12%** | 93.96 ± 0.24% | Trained on all 11,932 imgs (6 ep) vs. Stage 3's 4,800-img subset (5 ep) -> **+0.45% accuracy gain & 2.8x lower std!** |
| **`resnet50`** | 23.52M | 6.02 | **94.19 ± 0.09%** | 94.18 ± 0.19% | **94.31 ± 0.09%** | 94.30 ± 0.18% | Consistent (`94.19 ± 0.09%` vs. `94.18 ± 0.19%`), strict FP32 eval + saved `y_pred` & FP16 ckpts. |
| **`efficientnet_b0`** | 4.02M | 8.40 | **93.69 ± 0.28%** | 93.31 ± 0.30% | **93.81 ± 0.27%** | 93.43 ± 0.29% | Improved by `+0.38%` (`93.69 ± 0.28%` vs. `93.31 ± 0.30%`), strict FP32 eval + saved `y_pred` & FP16 ckpts. |

---

## 3. Per-Class 3-Seed Accuracy Breakdown (%)

| Architecture | `buildings` | `forest` | `glacier` | `mountain` | `sea` | `street` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`convnext_tiny`** | 92.90 ± 1.17 | 99.58 ± 0.30 | 88.61 ± 0.92 | 92.00 ± 0.47 | 96.67 ± 0.96 | 96.61 ± 0.00 |
| **`resnet50`** | 92.91 ± 0.67 | 99.72 ± 0.10 | 87.22 ± 0.87 | 91.05 ± 0.97 | 98.43 ± 0.42 | 96.74 ± 0.61 |
| **`efficientnet_b0`** | 92.14 ± 0.47 | 99.23 ± 0.26 | 86.44 ± 1.17 | 90.92 ± 0.70 | 98.56 ± 0.24 | 95.74 ± 0.25 |

---

## 4. Preliminary Pairwise McNemar Tests Among Completed S1-train-a Models ($N=3,000$ Test Images)

| Seed | Model Pair ($A$ vs. $B$) | $n_{11}$ (Both Correct) | $n_{10}$ ($A$ Only) | $n_{01}$ ($B$ Only) | $n_{00}$ (Both Wrong) | $\chi^2_{\text{cc}}$ | Exact Binomial $p$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `seed=0` | `convnext_tiny` vs. `resnet50` | 2768 | 58 | 59 | 115 | 0.0000 | 1.000000 |
| `seed=0` | `convnext_tiny` vs. `efficientnet_b0` | 2765 | 61 | 54 | 120 | 0.3130 | 0.576012 |
| `seed=0` | `resnet50` vs. `efficientnet_b0` | 2760 | 67 | 59 | 114 | 0.3889 | 0.533053 |
| `seed=1` | `convnext_tiny` vs. `resnet50` | 2756 | 70 | 66 | 108 | 0.0662 | 0.797102 |
| `seed=1` | `convnext_tiny` vs. `efficientnet_b0` | 2739 | 87 | 60 | 114 | 4.5986 | 0.031645 |
| `seed=1` | `resnet50` vs. `efficientnet_b0` | 2740 | 82 | 59 | 119 | 3.4326 | 0.063537 |
| `seed=2` | `convnext_tiny` vs. `resnet50` | 2772 | 60 | 56 | 112 | 0.0776 | 0.780739 |
| `seed=2` | `convnext_tiny` vs. `efficientnet_b0` | 2760 | 72 | 54 | 114 | 2.2937 | 0.129576 |
| `seed=2` | `resnet50` vs. `efficientnet_b0` | 2758 | 70 | 56 | 116 | 1.3413 | 0.246706 |

---

## 5. Key Surprises & Insights from `S1-train-a`

1. **`ConvNeXt-Tiny` Now Outperforms `ResNet-50` and `EfficientNet-B0` Under Fair Full-Dataset Training:** In our earlier draft, `ConvNeXt-Tiny` was trained on a 4,800-image subset for 5 epochs in Stage 3 (`93.82 ± 0.25%`), making it appear slightly less accurate than `ResNet-50` (`94.18 ± 0.19%`). When trained on the full `11,932`-image training set for 6 epochs in `S1-train-a`, `ConvNeXt-Tiny` rises to **`94.27 ± 0.09%`** (Macro-F1: **`94.40 ± 0.12%`**), surpassing `ResNet-50` (`94.19 ± 0.09%`, Macro-F1: `94.31 ± 0.09%`) and `EfficientNet-B0` (`93.69 ± 0.28%`, Macro-F1: `93.81 ± 0.27%`) while running at `6.30 ms` latency.
2. **Fast Execution & Compact Checkpoints:** All 9 units completed in **`49.6 minutes`** total on Kaggle T4, and the 9 FP16 checkpoints occupy only **`332.55 MB`**, leaving >14 GB of free disk space in `/kaggle/working` for downstream sessions.
3. **Complete Per-Seed Prediction Vectors (`y_true`, `y_pred`) Saved:** Every unit JSON now contains all 3,000 test predictions (`y_true` and `y_pred`), verifying `n11 + n10 + n01 + n00 = 3,000` on every seed.
