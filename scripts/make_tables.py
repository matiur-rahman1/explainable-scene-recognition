"""Generate LaTeX tables automatically from JSON experiment logs in results/."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

MODEL_DISPLAY = {
    "resnet50": "ResNet-50 (CNN)",
    "efficientnet_b0": "EfficientNet-B0 (CNN)",
    "swin_tiny": "Swin-Tiny (ViT)",
    "clip_zeroshot": "CLIP ViT-B/16 (Zero-Shot)",
    "clip_linear": "CLIP ViT-B/16 (Linear Probe)",
    "resnet50_pgd_at": "ResNet-50 + PGD-AT",
    "swin_tiny_pgd_at": "Swin-Tiny + PGD-AT",
}


def _fmt_ms(metric_dict: Dict[str, float], scale: float = 100.0, digits: int = 2) -> str:
    """Format {'mean': m, 'std': s} as 'm ± s' in LaTeX."""
    if not metric_dict or "mean" not in metric_dict:
        return "\\textbf{[TBD]}"
    m = metric_dict["mean"] * scale
    s = metric_dict.get("std", 0.0) * scale
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def generate_classification_table(multi_seed: Dict[str, Any], out_path: Path) -> None:
    """Write LaTeX table for multi-seed classification accuracy, F1, params, and latency."""
    agg = multi_seed.get("aggregated", {})
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Multi-Seed Classification Performance and Computational Efficiency (Mean $\\pm$ Std across Seeds).}",
        "\\label{tab:classification_main}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "\\textbf{Architecture} & \\textbf{Test Accuracy (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{Weighted-F1 (\\%)} & \\textbf{Params (M)} & \\textbf{Latency (ms)} \\\\",
        "\\hline",
    ]
    for m_key, m_data in agg.items():
        disp = MODEL_DISPLAY.get(m_key, m_key)
        acc = _fmt_ms(m_data.get("test_accuracy", {}), scale=100.0, digits=2)
        mf1 = _fmt_ms(m_data.get("macro_f1", {}), scale=100.0, digits=2)
        wf1 = _fmt_ms(m_data.get("weighted_f1", {}), scale=100.0, digits=2)
        stats = m_data.get("stats", {})
        params = f"{stats.get('total_params_m', 0.0):.2f}"
        lat = f"{stats.get('latency_ms', 0.0):.2f}"
        lines.append(f"{disp} & {acc} & {mf1} & {wf1} & {params} & {lat} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_xai_table(multi_seed: Dict[str, Any], out_path: Path) -> None:
    """Write LaTeX table for quantitative XAI faithfulness (Insertion and Deletion AUC)."""
    agg = multi_seed.get("aggregated", {})
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Quantitative XAI Faithfulness: Insertion AUC ($\\uparrow$) and Deletion AUC ($\\downarrow$) on Stratified Test Samples.}",
        "\\label{tab:xai_faithfulness}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Grad-CAM Ins$\\uparrow$} & \\textbf{Grad-CAM Del$\\downarrow$} & \\textbf{Grad-CAM++ Ins$\\uparrow$} & \\textbf{Grad-CAM++ Del$\\downarrow$} & \\textbf{LayerCAM Ins$\\uparrow$} & \\textbf{LayerCAM Del$\\downarrow$} \\\\",
        "\\hline",
    ]
    for m_key, m_data in agg.items():
        disp = MODEL_DISPLAY.get(m_key, m_key)
        xai = m_data.get("xai_faithfulness", {})
        cells = []
        for method in ("gradcam", "gradcam++", "layercam"):
            if method in xai:
                ins = _fmt_ms(xai[method].get("insertion_auc", {}), scale=1.0, digits=4)
                dele = _fmt_ms(xai[method].get("deletion_auc", {}), scale=1.0, digits=4)
            else:
                ins, dele = "--", "--"
            cells.extend([ins, dele])
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_robustness_table(multi_seed: Dict[str, Any], out_path: Path) -> None:
    """Write LaTeX table for FGSM and PGD adversarial robustness across epsilons."""
    agg = multi_seed.get("aggregated", {})
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Adversarial Robustness (Accuracy \\%) under FGSM and PGD-$L_\\infty$ Attacks on Class-Stratified Test Set.}",
        "\\label{tab:adversarial_robustness}",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Clean} & \\textbf{FGSM .01} & \\textbf{FGSM .03} & \\textbf{FGSM .05} & \\textbf{PGD .01} & \\textbf{PGD .03} & \\textbf{PGD .05} \\\\",
        "\\hline",
    ]
    for m_key, m_data in agg.items():
        disp = MODEL_DISPLAY.get(m_key, m_key)
        fgsm = m_data.get("fgsm_accuracy", {})
        pgd = m_data.get("pgd_accuracy", {})
        clean = _fmt_ms(fgsm.get("0.0", {}), scale=100.0, digits=2)
        f01 = _fmt_ms(fgsm.get("0.01", {}), scale=100.0, digits=2)
        f03 = _fmt_ms(fgsm.get("0.03", {}), scale=100.0, digits=2)
        f05 = _fmt_ms(fgsm.get("0.05", {}), scale=100.0, digits=2)
        p01 = _fmt_ms(pgd.get("0.01", {}), scale=100.0, digits=2)
        p03 = _fmt_ms(pgd.get("0.03", {}), scale=100.0, digits=2)
        p05 = _fmt_ms(pgd.get("0.05", {}), scale=100.0, digits=2)
        lines.append(
            f"{disp} & {clean} & {f01} & {f03} & {f05} & {p01} & {p03} & {p05} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="intel")
    parser.add_argument("--tables-dir", type=str, default="./paper/tables")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    tab_dir = Path(args.tables_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)

    ms_path = res_dir / f"multi_seed_{args.dataset}.json"
    if not ms_path.is_file():
        print(f"No {ms_path} found yet; skipping table generation.")
        return

    with open(ms_path, "r", encoding="utf-8") as f:
        ms_data = json.load(f)

    generate_classification_table(
        ms_data, tab_dir / f"tab_classification_{args.dataset}.tex"
    )
    generate_xai_table(ms_data, tab_dir / f"tab_xai_{args.dataset}.tex")
    generate_robustness_table(ms_data, tab_dir / f"tab_robustness_{args.dataset}.tex")
    print(f"Generated LaTeX tables in {tab_dir.resolve()}")


if __name__ == "__main__":
    main()
