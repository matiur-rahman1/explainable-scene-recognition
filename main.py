"""End-to-end multi-seed training, XAI faithfulness, CLIP, PGD-AT, and robustness runner."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
        choices=["intel", "sun397", "sun397_100", "custom"],
        help="Dataset to train and evaluate on.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Optional class limit (e.g. 100 for SUN397-100).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "efficientnet_b0", "swin_tiny"],
        help="Backbones to benchmark (e.g. resnet50 efficientnet_b0 swin_tiny clip_zeroshot clip_linear).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for multi-seed evaluation (e.g. --seeds 0 1 2).",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--xai-samples", type=int, default=60)
    parser.add_argument("--adv-samples", type=int, default=600)
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        default=["gradcam", "gradcam++", "layercam"],
    )
    parser.add_argument(
        "--pgd-at",
        action="store_true",
        help="Run PGD Adversarial Training (eps=0.03) on --pgd-at-models.",
    )
    parser.add_argument(
        "--pgd-at-models",
        nargs="+",
        default=["resnet50", "swin_tiny"],
    )
    parser.add_argument("--pgd-at-epochs", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed checkpoints in --checkpoint-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    res_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device} | Seeds: {args.seeds}")

    per_model_seed_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in args.models}
    canonical_seed = args.seeds[0]
    canonical_histories: Dict[str, dict] = {}
    canonical_fgsm: Dict[str, Dict[float, float]] = {}
    canonical_pgd: Dict[str, Dict[float, float]] = {}
    canonical_preds: Dict[str, List[int]] = {}
    canonical_y_true: List[int] = []
    zeroshot_cache: Dict[str, Dict[str, Any]] = {}

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'#' * 68}\nRUNNING SEED {seed} ({seed_idx + 1}/{len(args.seeds)})\n{'#' * 68}")
        set_seed(seed)

        cfg_path = str(res_dir / f"config_{args.dataset}_seed{seed}.json")
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=seed,
            max_classes=args.max_classes,
            config_save_path=cfg_path if seed_idx == 0 else None,
        )
        num_classes = len(class_names)
        if seed_idx == 0:
            # Also save canonical config.json
            with open(res_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": args.dataset,
                        "seeds": args.seeds,
                        "num_classes": num_classes,
                        "class_names": class_names,
                        "num_train": len(train_loader.dataset),
                        "num_val": len(val_loader.dataset),
                        "num_test": len(test_loader.dataset),
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "xai_samples": args.xai_samples,
                        "adv_samples": args.adv_samples,
                    },
                    f,
                    indent=2,
                )
            print(f"Loaded dataset '{args.dataset}' with {num_classes} classes.")

        sample_batch_imgs, sample_batch_labels = next(iter(test_loader))
        sample_img = sample_batch_imgs[0:1]
        sample_lbl = int(sample_batch_labels[0].item())

        for model_name in args.models:
            # Zero-shot CLIP is deterministic across seeds; reuse after first seed
            if model_name == "clip_zeroshot" and model_name in zeroshot_cache:
                print(f"[{model_name}] Reusing deterministic zero-shot evaluation for seed {seed}")
                per_model_seed_runs[model_name].append(zeroshot_cache[model_name])
                continue

            print(f"\n{'=' * 60}\nBenchmarking: {model_name} (seed={seed})\n{'=' * 60}")
            set_seed(seed)
            model = build_model(
                model_name,
                num_classes=num_classes,
                pretrained=True,
                class_names=class_names,
            )

            tag = f"{args.dataset}_seed{seed}" if len(args.seeds) > 1 else None
            model_epochs = 5 if model_name == "clip_linear" else args.epochs
            model_lr = 1e-3 if model_name == "clip_linear" else args.lr

            model, history, ckpt_path = train_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                epochs=model_epochs,
                lr=model_lr,
                checkpoint_dir=args.checkpoint_dir,
                tag=tag,
                resume=args.resume,
            )

            eval_res = evaluate_model(model, test_loader, device, class_names)
            stats = get_model_stats(model, device, img_size=args.img_size)
            print(
                f"[{model_name} | seed={seed}] Test Acc: {eval_res['accuracy']*100:.2f}% | "
                f"Macro-F1: {eval_res['macro_f1']*100:.2f}% | "
                f"Params: {stats['total_params_m']}M | Latency: {stats['latency_ms']} ms"
            )

            # Evaluate XAI faithfulness on seed 0 (or every seed on Intel)
            run_xai = (seed == canonical_seed) or (args.dataset == "intel")
            xai_faith = (
                evaluate_xai_faithfulness(
                    model,
                    model_name,
                    test_loader,
                    device,
                    methods=args.xai_methods,
                    max_samples=args.xai_samples,
                    seed=seed,
                )
                if run_xai
                else {}
            )
            if xai_faith:
                print(f"[{model_name} | seed={seed}] XAI Faithfulness: {xai_faith}")

            fgsm_res = evaluate_adversarial_robustness(
                model,
                test_loader,
                device,
                attack_type="fgsm",
                max_samples=args.adv_samples,
                seed=seed,
            )
            fgsm_pred_res = evaluate_adversarial_robustness(
                model,
                test_loader,
                device,
                attack_type="fgsm_pred",
                max_samples=args.adv_samples,
                seed=seed,
            )
            pgd_res = evaluate_adversarial_robustness(
                model,
                test_loader,
                device,
                attack_type="pgd",
                max_samples=args.adv_samples,
                seed=seed,
            )

            run_record = {
                "seed": seed,
                "checkpoint": str(ckpt_path),
                "test_accuracy": round(eval_res["accuracy"], 4),
                "macro_f1": round(eval_res["macro_f1"], 4),
                "weighted_f1": round(eval_res["weighted_f1"], 4),
                "per_class_accuracy": eval_res["per_class_accuracy"],
                "stats": stats,
                "xai_faithfulness": xai_faith,
                "fgsm_accuracy": fgsm_res,
                "fgsm_pred_accuracy": fgsm_pred_res,
                "pgd_accuracy": pgd_res,
            }
            per_model_seed_runs[model_name].append(run_record)
            if model_name == "clip_zeroshot":
                zeroshot_cache[model_name] = run_record

            if seed == canonical_seed:
                canonical_histories[model_name] = history
                canonical_fgsm[model_name] = fgsm_res
                canonical_pgd[model_name] = pgd_res
                canonical_preds[model_name] = eval_res["y_pred"].tolist()
                canonical_y_true = eval_res["y_true"].tolist()

                if num_classes <= 20:
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

    # Save canonical single-seed summary for backward compatibility
    canonical_summary = {
        m: runs[0] for m, runs in per_model_seed_runs.items() if runs
    }
    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(canonical_summary, f, indent=2)

    # Save multi-seed aggregated summary + raw per-seed runs
    multi_seed_summary = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "aggregated": {
            m: aggregate_seed_metrics(runs)
            for m, runs in per_model_seed_runs.items()
            if runs
        },
        "per_seed_runs": per_model_seed_runs,
    }
    with open(res_dir / f"multi_seed_{args.dataset}.json", "w", encoding="utf-8") as f:
        json.dump(multi_seed_summary, f, indent=2)
    with open(out_dir / f"multi_seed_{args.dataset}.json", "w", encoding="utf-8") as f:
        json.dump(multi_seed_summary, f, indent=2)

    # Compute pairwise McNemar statistical significance tests
    if len(canonical_preds) >= 2 and canonical_y_true:
        mcnemar_res = compute_pairwise_mcnemar(canonical_y_true, canonical_preds)
        with open(res_dir / f"mcnemar_{args.dataset}.json", "w", encoding="utf-8") as f:
            json.dump(mcnemar_res, f, indent=2)
        with open(out_dir / f"mcnemar_{args.dataset}.json", "w", encoding="utf-8") as f:
            json.dump(mcnemar_res, f, indent=2)

    plot_training_curves(
        canonical_histories, save_path=str(out_dir / "training_curves.png")
    )
    plot_adversarial_robustness(
        canonical_fgsm,
        attack_name="FGSM",
        save_path=str(out_dir / "robustness_fgsm.png"),
    )
    plot_adversarial_robustness(
        canonical_pgd,
        attack_name="PGD",
        save_path=str(out_dir / "robustness_pgd.png"),
    )

    # Optional Task A.5: PGD Adversarial Training (PGD-AT)
    if args.pgd_at:
        pgd_at_summary: Dict[str, Any] = {}
        set_seed(canonical_seed)
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=canonical_seed,
            max_classes=args.max_classes,
        )
        num_classes = len(class_names)
        sample_batch_imgs, sample_batch_labels = next(iter(test_loader))
        sample_img = sample_batch_imgs[0:1]
        sample_lbl = int(sample_batch_labels[0].item())

        for model_name in args.pgd_at_models:
            print(f"\n{'=' * 60}\nRunning PGD-AT (eps=0.03): {model_name}\n{'=' * 60}")
            model = build_model(
                model_name,
                num_classes=num_classes,
                pretrained=True,
                class_names=class_names,
            )
            # Load clean fine-tuned checkpoint as warm start if available
            tag_ckpt = Path(args.checkpoint_dir) / f"best_{model_name}_{args.dataset}_seed{canonical_seed}.pth"
            base_ckpt = Path(args.checkpoint_dir) / f"best_{model_name}.pth"
            if tag_ckpt.is_file():
                ckpt = torch.load(tag_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
            elif base_ckpt.is_file():
                ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])

            model, at_hist, at_ckpt = train_pgd_adversarial(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                epochs=args.pgd_at_epochs,
                epsilon=0.03,
                checkpoint_dir=args.checkpoint_dir,
            )

            eval_res = evaluate_model(model, test_loader, device, class_names)
            xai_faith = evaluate_xai_faithfulness(
                model,
                model_name,
                test_loader,
                device,
                methods=args.xai_methods,
                max_samples=args.xai_samples,
                seed=canonical_seed,
            )
            fgsm_res = evaluate_adversarial_robustness(
                model,
                test_loader,
                device,
                attack_type="fgsm",
                max_samples=args.adv_samples,
                seed=canonical_seed,
            )
            pgd_res = evaluate_adversarial_robustness(
                model,
                test_loader,
                device,
                attack_type="pgd",
                max_samples=args.adv_samples,
                seed=canonical_seed,
            )

            plot_xai_comparison_grid(
                model,
                model_name,
                test_loader,
                class_names,
                device,
                save_path=str(out_dir / f"xai_grid_{model_name}_pgd_at.png"),
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
                save_path=str(out_dir / f"adv_drift_{model_name}_pgd_at.png"),
            )

            pgd_at_summary[f"{model_name}_pgd_at"] = {
                "checkpoint": str(at_ckpt),
                "history": at_hist,
                "test_accuracy": round(eval_res["accuracy"], 4),
                "macro_f1": round(eval_res["macro_f1"], 4),
                "weighted_f1": round(eval_res["weighted_f1"], 4),
                "xai_faithfulness": xai_faith,
                "fgsm_accuracy": fgsm_res,
                "pgd_accuracy": pgd_res,
            }

        with open(res_dir / f"pgd_at_{args.dataset}.json", "w", encoding="utf-8") as f:
            json.dump(pgd_at_summary, f, indent=2)
        with open(out_dir / f"pgd_at_{args.dataset}.json", "w", encoding="utf-8") as f:
            json.dump(pgd_at_summary, f, indent=2)

    print(f"\nAll results saved to: {res_dir.resolve()} and {out_dir.resolve()}")


if __name__ == "__main__":
    main()
