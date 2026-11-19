"""Generate 300-DPI architectural pipeline and cross-paradigm comparison figures for the IEEE Access paper."""

import json
import shutil
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def draw_pipeline_overview(save_path: Path) -> None:
    """Render a clean 300-DPI schematic of the unified explainable & robust benchmark."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.3, 1.1, 2.1, 2.0, "#EFF6FF", "#1D4ED8", "1. Scene Datasets\n\n• Intel Scenes (6 cls)\n• SUN397-100 (100 cls)\n• Stratified Splits\n• Multi-Seed {0, 1, 2}"),
        (2.9, 1.1, 2.4, 2.0, "#F0FDF4", "#15803D", "2. Three Paradigms\n\n• CNN: ResNet-50,\n  EfficientNet-B0\n• ViT: Swin-Tiny\n• VLM: CLIP ViT-B/16\n  (Zero-Shot & Linear)"),
        (5.8, 1.1, 2.3, 2.0, "#FEF3C7", "#B45309", "3. Gradient XAI\n\n• Grad-CAM\n• Grad-CAM++\n• LayerCAM\n• 2D Token Reshaping\n• Insertion/Deletion AUC"),
        (8.6, 1.1, 2.1, 2.0, "#FEE2E2", "#B91C1C", "4. Robustness\n\n• FGSM & FGSM-Pred\n• 10-Step PGD-L∞\n• Attention Drift\n• Mixed PGD-AT"),
        (11.2, 1.1, 1.5, 2.0, "#F3E8FF", "#6D28D9", "5. Evaluation\n\n• Acc / Macro-F1\n• Latency / Params\n• McNemar Test\n• Failure Modes"),
    ]

    for x, y, w, h, facecolor, edgecolor, text in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            linewidth=1.8,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#111827",
        )

    arrow_xs = [(2.45, 2.85), (5.35, 5.75), (8.15, 8.55), (10.75, 11.15)]
    for x0, x1 in arrow_xs:
        ax.annotate(
            "",
            xy=(x1, 2.1),
            xytext=(x0, 2.1),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#374151"),
        )

    ax.set_title(
        "Unified Explainable & Adversarial Scene Recognition Benchmark Pipeline",
        fontsize=12.5,
        fontweight="bold",
        color="#1B365D",
        pad=10,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_paradigm_comparison(summary_json_path: Path, save_path: Path) -> None:
    """Plot a 300-DPI grouped bar chart comparing Accuracy, Faithfulness, and PGD Robustness."""
    if not summary_json_path.is_file():
        return

    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agg = data.get("aggregated", data)
    models = list(agg.keys())
    if not models:
        return

    labels = []
    acc_vals = []
    ins_vals = []
    pgd_vals = []

    display_map = {
        "resnet50": "ResNet-50",
        "efficientnet_b0": "EfficientNet-B0",
        "swin_tiny": "Swin-Tiny",
        "clip_zeroshot": "CLIP (Zero-Shot)",
        "clip_linear": "CLIP (Linear)",
    }

    for m in models:
        m_info = agg[m]
        labels.append(display_map.get(m, m))

        ta = m_info.get("test_accuracy", 0.0)
        acc = (ta["mean"] if isinstance(ta, dict) else float(ta)) * 100.0
        acc_vals.append(acc)

        xai = m_info.get("xai_faithfulness", {})
        best_ins = 0.0
        for method_dict in xai.values():
            ia = method_dict.get("insertion_auc", 0.0)
            val = (ia["mean"] if isinstance(ia, dict) else float(ia)) * 100.0
            best_ins = max(best_ins, val)
        ins_vals.append(best_ins)

        pgd = m_info.get("pgd_accuracy", {})
        p03 = pgd.get("0.03", 0.0)
        pval = (p03["mean"] if isinstance(p03, dict) else float(p03)) * 100.0
        pgd_vals.append(pval)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width, acc_vals, width, label="Clean Test Accuracy (%)", color="#1D4ED8")
    ax.bar(x, ins_vals, width, label="Best XAI Insertion AUC (×100)", color="#15803D")
    ax.bar(x + width, pgd_vals, width, label="PGD Robustness (ε=0.03, %)", color="#B91C1C")

    ax.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "Cross-Paradigm Comparison: Clean Accuracy vs. XAI Faithfulness vs. PGD Robustness",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    draw_pipeline_overview(fig_dir / "pipeline_overview.png")

    # Copy existing 300-DPI figures from assets/ or outputs/
    for src_folder in (root / "assets", root / "outputs"):
        if src_folder.is_dir():
            for png in src_folder.glob("*.png"):
                shutil.copy2(png, fig_dir / png.name)

    ms_json = root / "results" / "multi_seed_intel.json"
    if ms_json.is_file():
        draw_paradigm_comparison(ms_json, fig_dir / "clip_cnn_vit_comparison.png")
    elif (root / "assets" / "benchmark_summary.json").is_file():
        draw_paradigm_comparison(
            root / "assets" / "benchmark_summary.json",
            fig_dir / "clip_cnn_vit_comparison.png",
        )

    print(f"All 300-DPI paper figures prepared in: {fig_dir}")


if __name__ == "__main__":
    main()
