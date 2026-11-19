"""Stage 2 Runner with Per-Model Checkpoint Resume & Incremental Zip Saving."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.adversarial import evaluate_adversarial_robustness, train_pgd_adversarial
from src.dataset import TransformSubset, create_dataloaders
from src.models import build_model, get_model_stats
from src.stats import aggregate_seed_metrics, compute_pairwise_mcnemar
from src.trainer import evaluate_model, train_model
from src.utils import get_stratified_indices, set_seed
from src.visualize import (
    plot_adversarial_attention_drift,
    plot_adversarial_robustness,
    plot_training_curves,
    plot_xai_comparison_grid,
)
from src.xai import evaluate_xai_faithfulness


def _save_incremental_zips(res_dir: Path, out_dir: Path) -> None:
    """Refresh /kaggle/working zip archives after every model so progress is never lost."""
    if Path("/kaggle/working").is_dir():
        shutil.make_archive("/kaggle/working/stage2_json_results", "zip", str(res_dir))
        shutil.make_archive("/kaggle/working/stage2_figures", "zip", str(out_dir))


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

    # Use a 3,600-image class-balanced stratified training subset (600/class) for fast PGD-AT epochs
    train_ds = train_loader.dataset
    sub_idx = get_stratified_indices(train_ds.targets, max_samples=3600, seed=seed)
    fast_train_ds = TransformSubset(train_ds, sub_idx, transform=None, raw_targets=train_ds.targets)
    fast_train_loader = DataLoader(
        fast_train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    sample_imgs, sample_lbls = next(iter(test_loader))
    sample_img = sample_imgs[0:1]
    sample_lbl = int(sample_lbls[0].item())

    pgd_json_path = res_dir / "pgd_at_intel.json"
    pgd_at_summary: Dict[str, Any] = {}
    if pgd_json_path.is_file():
        with open(pgd_json_path, "r", encoding="utf-8") as f:
            pgd_at_summary = json.load(f)

    for model_name in ("resnet50", "swin_tiny"):
        key = f"{model_name}_pgd_at"
        if key in pgd_at_summary:
            print(f"[{key}] Already evaluated in {pgd_json_path}; skipping!")
            continue

        print(f"\n{'=' * 60}\nPGD-AT (eps=0.03): {model_name}\n{'=' * 60}")
        set_seed(seed)
        model = build_model(
            model_name, num_classes=num_classes, pretrained=True, class_names=class_names
        )

        at_ckpt_path = checkpoint_dir / f"best_{model_name}_pgd_at.pth"
        if at_ckpt_path.is_file():
            print(f"[{model_name}] Found existing PGD-AT checkpoint {at_ckpt_path}! Loading and running 1 polish epoch...")
            ckpt = torch.load(at_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            at_epochs = 1
        else:
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
                print(f"[{model_name}] Running 2 clean warm-up epochs before PGD-AT...")
                model, _, _ = train_model(
                    model=model,
                    model_name=model_name,
                    train_loader=fast_train_loader,
                    val_loader=val_loader,
                    class_names=class_names,
                    device=device,
                    epochs=2,
                    lr=2e-4,
                    checkpoint_dir=str(checkpoint_dir),
                )
            at_epochs = 3

        model, at_hist, at_ckpt = train_pgd_adversarial(
            model=model,
            model_name=model_name,
            train_loader=fast_train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            epochs=at_epochs,
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

        pgd_at_summary[key] = {
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

        with open(pgd_json_path, "w", encoding="utf-8") as f:
            json.dump(pgd_at_summary, f, indent=2)
        with open(out_dir / "pgd_at_intel.json", "w", encoding="utf-8") as f:
            json.dump(pgd_at_summary, f, indent=2)
        _save_incremental_zips(res_dir, out_dir)


def run_sun397_100(
    device: torch.device,
    checkpoint_dir: Path,
    out_dir: Path,
    res_dir: Path,
    seed: int = 0,
) -> None:
    """Run the 100-class SUN397 protocol across all 5 architectures with per-model resume."""
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
    sun_json_path = res_dir / "multi_seed_sun397_100.json"
    per_model_runs: Dict[str, list] = {m: [] for m in models}
    if sun_json_path.is_file():
        with open(sun_json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            for m, runs in existing.get("per_seed_runs", {}).items():
                if runs:
                    per_model_runs[m] = runs

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
        epochs = 5 if model_name == "clip_linear" else 5
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
        preds_by_model[model_name] = eval_res["y_pred"].tolist()
        y_true_list = eval_res["y_true"].tolist()

        if per_model_runs.get(model_name):
            print(f"[SUN397-100 | {model_name}] Reusing saved XAI & robustness record from JSON!")
            record = per_model_runs[model_name][0]
            fgsm_curves[model_name] = {float(k): float(v) for k, v in record["fgsm_accuracy"].items()}
            pgd_curves[model_name] = {float(k): float(v) for k, v in record["pgd_accuracy"].items()}
            continue

        stats = get_model_stats(model, device, img_size=224)
        print(
            f"[SUN397-100 | {model_name}] Test Acc: {eval_res['accuracy']*100:.2f}% | "
            f"Macro-F1: {eval_res['macro_f1']*100:.2f}%"
        )

        xai_faith = evaluate_xai_faithfulness(
            model,
            model_name,
            test_loader,
            device,
            methods=("gradcam", "gradcam++", "layercam"),
            max_samples=200,
            steps=10,
            seed=seed,
        )
        print(f"[SUN397-100 | {model_name}] XAI Faithfulness: {xai_faith}")

        fgsm_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="fgsm", max_samples=500, seed=seed
        )
        pgd_res = evaluate_adversarial_robustness(
            model, test_loader, device, attack_type="pgd", max_samples=500, seed=seed
        )

        fgsm_curves[model_name] = fgsm_res
        pgd_curves[model_name] = pgd_res

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
        per_model_runs[model_name] = [record]

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
            "aggregated": {
                m: aggregate_seed_metrics(runs)
                for m, runs in per_model_runs.items()
                if runs
            },
            "per_seed_runs": per_model_runs,
        }
        with open(sun_json_path, "w", encoding="utf-8") as f:
            json.dump(sun_summary, f, indent=2)
        with open(out_dir / "multi_seed_sun397_100.json", "w", encoding="utf-8") as f:
            json.dump(sun_summary, f, indent=2)
        _save_incremental_zips(res_dir, out_dir)

    if len(preds_by_model) >= 2 and y_true_list:
        mcnemar_sun = compute_pairwise_mcnemar(y_true_list, preds_by_model)
        with open(res_dir / "mcnemar_sun397_100.json", "w", encoding="utf-8") as f:
            json.dump(mcnemar_sun, f, indent=2)

    if histories:
        plot_training_curves(histories, save_path=str(out_dir / "sun397_training_curves.png"))
    if fgsm_curves:
        plot_adversarial_robustness(
            fgsm_curves,
            attack_name="FGSM (SUN397-100)",
            save_path=str(out_dir / "sun397_robustness_fgsm.png"),
        )
    if pgd_curves:
        plot_adversarial_robustness(
            pgd_curves,
            attack_name="PGD (SUN397-100)",
            save_path=str(out_dir / "sun397_robustness_pgd.png"),
        )
    _save_incremental_zips(res_dir, out_dir)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("./checkpoints")
    out_dir = Path("./outputs")
    res_dir = Path("./results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    run_pgd_at_intel(device, ckpt_dir, out_dir, res_dir, seed=0)
    run_sun397_100(device, ckpt_dir, out_dir, res_dir, seed=0)
    print("\nSTAGE 2 COMPLETE!")


if __name__ == "__main__":
    main()
