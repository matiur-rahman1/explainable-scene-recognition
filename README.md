# Explainable Multi-Class Scene Recognition from Natural Images Using Deep Learning and Gradient-Based Visual Explanation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matiur-rahman1/explainable-scene-recognition/blob/main/notebooks/explainable_scene_recognition_colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A modular, reproducible PyTorch research framework comparing **Convolutional Neural Networks (`ResNet-50`, `EfficientNet-B0`)** and **Vision Transformers (`Swin-Tiny`)** for multi-class natural scene recognition, paired with **gradient-based visual explanations (`Grad-CAM`, `Grad-CAM++`, `LayerCAM`)**, **quantitative XAI faithfulness evaluation (`Insertion/Deletion AUC`)**, and **adversarial robustness analysis (`FGSM`, `PGD`)**.

---

## Research Objectives & Alignment

1. **CNN vs. Vision Transformer Benchmarking:** Systematic comparison of `ResNet-50`, `EfficientNet-B0`, and `Swin-Tiny` across accuracy, macro-F1, parameter count, and inference latency.
2. **Gradient-Based Visual Explainability (XAI):** Multi-method activation mapping (`Grad-CAM`, `Grad-CAM++`, `LayerCAM`) with architecture-aware feature reshaping for Swin Transformer token grids.
3. **Quantitative XAI Faithfulness:** Progressive pixel perturbation evaluation via **Insertion AUC** (↑ higher is better) and **Deletion AUC** (↓ lower is better) using a blurred baseline.
4. **Adversarial Testing & Failure Analysis:** Evaluation under **FGSM** and **PGD-$L_\infty$** perturbations ($\epsilon \in \{0.0, 0.01, 0.03, 0.05\}$), including **attention-drift visualization** and misclassified edge-case inspection.
5. **Interactive Demo:** Live **Gradio** web interface for real-time scene classification and heatmap inspection.

---

## Project Structure

```text
explainable-scene-recognition/
├── app.py                                           # Interactive Gradio demo
├── main.py                                          # End-to-end benchmark & XAI pipeline
├── requirements.txt                                 # Minimal runtime dependencies
├── notebooks/
│   └── explainable_scene_recognition_colab.ipynb    # One-click Google Colab GPU notebook
└── src/
    ├── __init__.py
    ├── dataset.py                                   # Intel Natural Scenes / SUN397 / Custom loader
    ├── models.py                                    # timm model factory & CAM target layer hooks
    ├── trainer.py                                   # AMP training loop, checkpointing & metrics
    ├── xai.py                                       # Grad-CAM, Grad-CAM++, LayerCAM & Insertion/Deletion AUC
    ├── adversarial.py                               # FGSM & PGD attacks in normalized pixel space
    └── visualize.py                                 # 300-DPI publication-quality figures
```

---

## Quick Start (Google Colab — Recommended)

Click the **Open In Colab** badge above (with **T4 GPU** enabled) to run the full pipeline—training, XAI evaluation, adversarial robustness testing, and the live Gradio demo—without installing anything on your local machine.

---

## Local / CLI Usage

```bash
# 1. Install dependencies in a virtual environment
pip install -r requirements.txt

# 2. Run full benchmark (Intel Natural Scenes: ResNet-50 vs. EfficientNet-B0 vs. Swin-Tiny)
python main.py --dataset intel --models resnet50 efficientnet_b0 swin_tiny --epochs 8 --batch-size 32

# 3. Optional: Scale up to SUN397 or a custom ImageFolder dataset
python main.py --dataset sun397 --epochs 10

# 4. Launch the interactive Gradio demo
python app.py
```

---

## Generated Outputs (`outputs/`)

Running `main.py` automatically saves 300-DPI figures and quantitative summaries:

- `benchmark_summary.json` — Accuracy, Macro-F1, Parameters (M), Latency (ms), Insertion/Deletion AUC, and FGSM/PGD accuracy curves.
- `training_curves.png` — Validation accuracy and loss comparison across models.
- `cm_<model>.png` — Per-model confusion matrix on the held-out test set.
- `xai_grid_<model>.png` — Side-by-side `Original | Grad-CAM | Grad-CAM++ | LayerCAM` explanations.
- `robustness_fgsm.png` & `robustness_pgd.png` — CNN vs. Transformer accuracy degradation curves.
- `adv_drift_<model>.png` — Clean vs. FGSM adversarial Grad-CAM attention shift.
- `failures_<model>.png` — Misclassified test cases with visual explanations.

---

## Author

**MD Matiur Rahman**  
BEng Computer Science & Technology, Nantong University, China (2026)  
GitHub: [github.com/matiur-rahman1](https://github.com/matiur-rahman1) | Email: matiurrahman1353@gmail.com
