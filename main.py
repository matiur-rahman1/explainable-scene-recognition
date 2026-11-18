"""End-to-end training, XAI faithfulness, and adversarial robustness benchmark runner."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from src.adversarial import evaluate_adversarial_robustness
from src.dataset import create_dataloaders
from src.models import build_model, get_model_stats
from src.trainer import evaluate_model, train_model
from src.visualize import (
    plot_adversarial_attention_drift,
    plot_adversarial_robustness,
    plot_confusion_matrix,
    plot_failure_cases,
    plot_training_curves,
    plot_xai_comparison_grid,
)
from src.xai import evaluate_xai_faithfulness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explainable Multi-Class Scene Recognition Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="intel",
        choices=["intel", "sun397", "custom"],
        help="Dataset to train and evaluate on.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "efficientnet_b0", "swin_tiny"],
        help="Backbones to benchmark (CNN vs. Transformer).",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--xai-samples", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)
    print(f"Loaded dataset '{args.dataset}' with {num_classes} classes: {class_names}")

    histories: Dict[str, dict] = {}
    fgsm_results: Dict[str, Dict[float, float]] = {}
    pgd_results: Dict[str, Dict[float, float]] = {}
    summary: Dict[str, dict] = {}

    sample_batch_imgs, sample_batch_labels = next(iter(test_loader))
    sample_img = sample_batch_imgs[0:1]
    sample_lbl = int(sample_batch_labels[0].item())

    for model_name in args.models:
        print(f"\n{'=' * 60}\nBenchmarking: {model_name}\n{'=' * 60}")
        model = build_model(model_name, num_classes=num_classes, pretrained=True)
        model, history, ckpt_path = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )
        histories[model_name] = history

        eval_res = evaluate_model(model, test_loader, device, class_names)
        stats = get_model_stats(model, device, img_size=args.img_size)
        print(
            f"[{model_name}] Test Acc: {eval_res['accuracy']*100:.2f}% | "
            f"Macro-F1: {eval_res['macro_f1']*100:.2f}% | "
            f"Params: {stats['total_params_m']}M | Latency: {stats['latency_ms']} ms"
        )

        xai_faith = evaluate_xai_faithfulness(
            model, model_name, test_loader, device, max_samples=args.xai_samples
        )
        print(f"[{model_name}] XAI Faithfulness: {xai_faith}")

        fgsm_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="fgsm"
        )
        pgd_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="pgd"
        )
        fgsm_results[model_name] = fgsm_res
        pgd_results[model_name] = pgd_res

        plot_confusion_matrix(
            eval_res["confusion_matrix"],
            class_names,
            title=f"Confusion Matrix ({model_name})",
            save_path=str(out_dir / f"cm_{model_name}.png"),
        )
        plot_xai_comparison_grid(
            model,
            model_name,
            test_loader,
            class_names,
            device,
            save_path=str(out_dir / f"xai_grid_{model_name}.png"),
            num_samples=min(4, num_classes),
        )
        plot_adversarial_attention_drift(
            model,
            model_name,
            sample_img,
            sample_lbl,
            class_names,
            device,
            epsilon=0.03,
            save_path=str(out_dir / f"adv_drift_{model_name}.png"),
        )
        plot_failure_cases(
            model,
            model_name,
            test_loader,
            class_names,
            device,
            save_path=str(out_dir / f"failures_{model_name}.png"),
            max_cases=4,
        )

        summary[model_name] = {
            "checkpoint": str(ckpt_path),
            "test_accuracy": round(eval_res["accuracy"], 4),
            "macro_f1": round(eval_res["macro_f1"], 4),
            "weighted_f1": round(eval_res["weighted_f1"], 4),
            "stats": stats,
            "xai_faithfulness": xai_faith,
            "fgsm_accuracy": fgsm_res,
            "pgd_accuracy": pgd_res,
        }

    plot_training_curves(histories, save_path=str(out_dir / "training_curves.png"))
    plot_adversarial_robustness(
        fgsm_results, attack_name="FGSM", save_path=str(out_dir / "robustness_fgsm.png")
    )
    plot_adversarial_robustness(
        pgd_results, attack_name="PGD", save_path=str(out_dir / "robustness_pgd.png")
    )

    summary_path = out_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results and figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
