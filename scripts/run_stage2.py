"""Stage 2 Runner: (1) PGD-AT (eps=0.03) on Intel Natural Scenes, and (2) 100-Class SUN397 Benchmark."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.adversarial import evaluate_adversarial_robustness, train_pgd_adversarial
from src.dataset import create_dataloaders
from src.models import build_model, get_model_stats
from src.stats import aggregate_seed_metrics, compute_pairwise_mcnemar
from src.trainer import evaluate_model, train_model
from src.utils import set_seed
from src.visualize import (
    plot_adversarial_attention_drift,
    plot_adversarial_robustness,
    plot_training_curves,
    plot_xai_comparison_grid,
)
from src.xai import evaluate_xai_faithfulness


def run_pgd_at_intel(
    device: torch.device,
    checkpoint_dir: Path,
    out_dir: Path,
    res_dir: Path,
    seed: int = 0,
) -> None:
    """Fine-tune ResNet-50 and Swin-Tiny with PGD-AT (eps=0.03) on Intel Natural Scenes."""
    print(f"\n{'#' * 68}\nSTAGE 2A: PGD ADVERSARIAL TRAINING (eps=0.03) ON INTEL SCENES\n{'#' * 68}")
    set_seed(seed)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_name="intel",
        img_size=224,
        batch_size=32,
        seed=seed,
    )
    num_classes = len(class_names)
    sample_imgs, sample_lbls = next(iter(test_loader))
    sample_img = sample_imgs[0:1]
    sample_lbl = int(sample_lbls[0].item())

    pgd_at_summary: Dict[str, Any] = {}
    for model_name in ("resnet50", "swin_tiny"):
        print(f"\n{'=' * 60}\nPGD-AT (eps=0.03): {model_name}\n{'=' * 60}")
        set_seed(seed)
        model = build_model(
            model_name, num_classes=num_classes, pretrained=True, class_names=class_names
        )

        # Warm-start from Stage 1 checkpoint if present, otherwise run 3 clean warm-up epochs
        candidates = [
            checkpoint_dir / f"best_{model_name}_intel_seed{seed}.pth",
            checkpoint_dir / f"best_{model_name}.pth",
        ]
        loaded = False
        for cpath in candidates:
            if cpath.is_file():
                print(f"[{model_name}] Warm-starting PGD-AT from {cpath}")
                ckpt = torch.load(cpath, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                loaded = True
                break
        if not loaded:
            print(f"[{model_name}] Running 3 clean warm-up epochs before PGD-AT...")
            model, _, _ = train_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                epochs=3,
                lr=2e-4,
                checkpoint_dir=str(checkpoint_dir),
            )

        model, at_hist, at_ckpt = train_pgd_adversarial(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            epochs=3,
            lr=1e-4,
            epsilon=0.03,
            pgd_steps=2,
            clean_weight=0.5,
            checkpoint_dir=str(checkpoint_dir),
        )

        eval_res = evaluate_model(model, test_loader, device, class_names)
        xai_faith = evaluate_xai_faithfulness(
            model, model_name, test_loader, device, max_samples=60, steps=20, seed=seed
        )
        fgsm_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="fgsm", max_samples=600, seed=seed
        )
        pgd_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="pgd", max_samples=600, seed=seed
        )

        print(
            f"[{model_name}+PGD-AT] Test Clean Acc: {eval_res['accuracy']*100:.2f}% | "
            f"Macro-F1: {eval_res['macro_f1']*100:.2f}% | "
            f"PGD(0.03): {pgd_res.get(0.03, 0.0)*100:.2f}% | "
            f"XAI: {xai_faith}"
        )

        plot_xai_comparison_grid(
            model,
            model_name,
            test_loader,
            class_names,
            device,
            save_path=str(out_dir / f"xai_grid_{model_name}_pgd_at.png"),
            num_samples=4,
        )
        plot_adversarial_attention_drift(
            model,
            model_name,
            sample_img,
            sample_lbl,
            class_names,
            device,
            epsilon=0.03,
            save_path=str(out_dir / f"adv_drift_{model_name}_pgd_at.png"),
        )

        pgd_at_summary[f"{model_name}_pgd_at"] = {
            "checkpoint": str(at_ckpt),
            "history": at_hist,
            "test_accuracy": round(eval_res["accuracy"], 4),
            "macro_f1": round(eval_res["macro_f1"], 4),
            "weighted_f1": round(eval_res["weighted_f1"], 4),
            "per_class_accuracy": eval_res["per_class_accuracy"],
            "xai_faithfulness": xai_faith,
            "fgsm_accuracy": fgsm_res,
            "pgd_accuracy": pgd_res,
        }

    with open(res_dir / "pgd_at_intel.json", "w", encoding="utf-8") as f:
        json.dump(pgd_at_summary, f, indent=2)
    with open(out_dir / "pgd_at_intel.json", "w", encoding="utf-8") as f:
        json.dump(pgd_at_summary, f, indent=2)


def run_sun397_100(
    device: torch.device,
    checkpoint_dir: Path,
    out_dir: Path,
    res_dir: Path,
    seed: int = 0,
) -> None:
    """Run the 100-class SUN397 protocol across all 5 architectures."""
    print(f"\n{'#' * 68}\nSTAGE 2B: 100-CLASS SUN397 LARGE-SCALE PROTOCOL (SEED={seed})\n{'#' * 68}")
    set_seed(seed)
    cfg_path = str(res_dir / "config_sun397_100.json")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_name="sun397_100",
        img_size=224,
        batch_size=32,
        seed=seed,
        max_classes=100,
        config_save_path=cfg_path,
    )
    num_classes = len(class_names)
    print(
        f"Loaded SUN397-100: {num_classes} classes | "
        f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )

    models = ["resnet50", "efficientnet_b0", "swin_tiny", "clip_zeroshot", "clip_linear"]
    per_model_runs: Dict[str, list] = {m: [] for m in models}
    histories: Dict[str, dict] = {}
    fgsm_curves: Dict[str, Dict[float, float]] = {}
    pgd_curves: Dict[str, Dict[float, float]] = {}
    preds_by_model: Dict[str, list] = {}
    y_true_list: list = []

    for model_name in models:
        print(f"\n{'=' * 60}\nBenchmarking on SUN397-100: {model_name}\n{'=' * 60}")
        set_seed(seed)
        model = build_model(
            model_name, num_classes=num_classes, pretrained=True, class_names=class_names
        )
        epochs = 5 if model_name == "clip_linear" else 6
        lr = 1e-3 if model_name == "clip_linear" else 2e-4

        model, hist, ckpt_path = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            epochs=epochs,
            lr=lr,
            checkpoint_dir=str(checkpoint_dir),
            tag=f"sun397_100_seed{seed}",
            resume=True,
        )
        histories[model_name] = hist

        eval_res = evaluate_model(model, test_loader, device, class_names)
        stats = get_model_stats(model, device, img_size=224)
        print(
            f"[SUN397-100 | {model_name}] Test Acc: {eval_res['accuracy']*100:.2f}% | "
            f"Macro-F1: {eval_res['macro_f1']*100:.2f}%"
        )

        # Evaluate XAI faithfulness and FGSM/PGD on fixed 500-image class-balanced test subset (5 per class)
        xai_faith = evaluate_xai_faithfulness(
            model,
            model_name,
            test_loader,
            device,
            methods=("gradcam", "gradcam++", "layercam"),
            max_samples=500,
            steps=10,
            seed=seed,
        )
        print(f"[SUN397-100 | {model_name}] XAI Faithfulness (500 imgs): {xai_faith}")

        fgsm_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="fgsm", max_samples=500, seed=seed
        )
        pgd_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="pgd", max_samples=500, seed=seed
        )

        fgsm_curves[model_name] = fgsm_res
        pgd_curves[model_name] = pgd_res
        preds_by_model[model_name] = eval_res["y_pred"].tolist()
        y_true_list = eval_res["y_true"].tolist()

        record = {
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "test_accuracy": round(eval_res["accuracy"], 4),
            "macro_f1": round(eval_res["macro_f1"], 4),
            "weighted_f1": round(eval_res["weighted_f1"], 4),
            "stats": stats,
            "xai_faithfulness": xai_faith,
            "fgsm_accuracy": fgsm_res,
            "pgd_accuracy": pgd_res,
        }
        per_model_runs[model_name].append(record)

        plot_xai_comparison_grid(
            model,
            model_name,
            test_loader,
            class_names,
            device,
            save_path=str(out_dir / f"sun397_xai_grid_{model_name}.png"),
            num_samples=4,
        )

    sun_summary = {
        "dataset": "sun397_100",
        "seeds": [seed],
        "aggregated": {m: aggregate_seed_metrics(runs) for m, runs in per_model_runs.items()},
        "per_seed_runs": per_model_runs,
    }
    with open(res_dir / "multi_seed_sun397_100.json", "w", encoding="utf-8") as f:
        json.dump(sun_summary, f, indent=2)
    with open(out_dir / "multi_seed_sun397_100.json", "w", encoding="utf-8") as f:
        json.dump(sun_summary, f, indent=2)

    mcnemar_sun = compute_pairwise_mcnemar(y_true_list, preds_by_model)
    with open(res_dir / "mcnemar_sun397_100.json", "w", encoding="utf-8") as f:
        json.dump(mcnemar_sun, f, indent=2)

    plot_training_curves(histories, save_path=str(out_dir / "sun397_training_curves.png"))
    plot_adversarial_robustness(
        fgsm_curves, attack_name="FGSM (SUN397-100)", save_path=str(out_dir / "sun397_robustness_fgsm.png")
    )
    plot_adversarial_robustness(
        pgd_curves, attack_name="PGD (SUN397-100)", save_path=str(out_dir / "sun397_robustness_pgd.png")
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("./checkpoints")
    out_dir = Path("./outputs")
    res_dir = Path("./results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    run_pgd_at_intel(device, ckpt_dir, out_dir, res_dir, seed=0)
    shutil.make_archive("/kaggle/working/stage2a_pgd_at_only", "zip", str(res_dir))

    run_sun397_100(device, ckpt_dir, out_dir, res_dir, seed=0)
    print("\nSTAGE 2 COMPLETE!")


if __name__ == "__main__":
    main()
