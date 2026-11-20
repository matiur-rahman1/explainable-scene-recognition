"""Stage 3 Neurocomputing Runner: ConvNeXt-Tiny, VG-LayerCAM, and SAC-PGD-AT Ablation Study."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.adversarial import (
    evaluate_adversarial_cam_similarity,
    evaluate_adversarial_robustness,
    fgsm_attack,
    train_sac_pgd_adversarial,
)
from src.dataset import TransformSubset, create_dataloaders, denormalize_image
from src.models import build_model, get_model_stats
from src.stats import aggregate_seed_metrics, compute_pairwise_mcnemar
from src.trainer import evaluate_model, train_model
from src.utils import get_stratified_indices, set_seed
from src.visualize import plot_confusion_matrix, plot_xai_comparison_grid
from src.xai import evaluate_xai_faithfulness, generate_cam_explanation


def _save_zips(res_dir: Path, out_dir: Path) -> None:
    if Path("/kaggle/working").is_dir():
        shutil.make_archive("/kaggle/working/stage3_neuro_json", "zip", str(res_dir))
        shutil.make_archive("/kaggle/working/stage3_neuro_figures", "zip", str(out_dir))


def plot_sac_drift_comparison(
    clean_model: torch.nn.Module,
    pgd_at_model: torch.nn.Module,
    sac_at_model: torch.nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    label_idx: int,
    class_names: List[str],
    device: torch.device,
    epsilon: float = 0.03,
    save_path: str = "outputs/adv_drift_sac_pgd_at.png",
) -> None:
    """Compare Clean Image | Clean Model Adv CAM | Standard PGD-AT Adv CAM | SAC-PGD-AT Adv CAM (Post-SAC-PGD-AT)."""
    x = image_tensor.to(device)
    y = torch.tensor([label_idx], device=device)

    adv_clean_m = fgsm_attack(clean_model, x, y, epsilon=epsilon)
    adv_pgd_m = fgsm_attack(pgd_at_model, x, y, epsilon=epsilon)
    adv_sac_m = fgsm_attack(sac_at_model, x, y, epsilon=epsilon)

    clean_rgb = denormalize_image(x[0])
    _, cam_clean_orig, p0, pr0 = generate_cam_explanation(
        clean_model, model_name, x, clean_rgb, method="vg_layercam"
    )
    _, cam_unprotected_adv, p1, pr1 = generate_cam_explanation(
        clean_model, model_name, adv_clean_m, denormalize_image(adv_clean_m[0]), method="vg_layercam"
    )
    _, cam_pgd_adv, p2, pr2 = generate_cam_explanation(
        pgd_at_model, model_name, adv_pgd_m, denormalize_image(adv_pgd_m[0]), method="vg_layercam"
    )
    _, cam_sac_adv, p3, pr3 = generate_cam_explanation(
        sac_at_model, model_name, adv_sac_m, denormalize_image(adv_sac_m[0]), method="vg_layercam"
    )

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    panels = [
        (cam_clean_orig, f"Clean Reference CAM\nPred: {class_names[p0]} ({pr0*100:.1f}%)"),
        (cam_unprotected_adv, f"Standard Model (Adv ε={epsilon})\nPred: {class_names[p1]} ({pr1*100:.1f}%)"),
        (cam_pgd_adv, f"Standard PGD-AT (β=0, Adv)\nPred: {class_names[p2]} ({pr2*100:.1f}%)"),
        (cam_sac_adv, f"SAC-PGD-AT (β=0.1, Post-SAC-PGD-AT)\nPred: {class_names[p3]} ({pr3*100:.1f}%)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Adversarial Attention-Drift Suppression ({model_name}, Post-SAC-PGD-AT)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_intel_convnext_and_vglayercam(
    device: torch.device, ckpt_dir: Path, out_dir: Path, res_dir: Path
) -> None:
    """Run 3-seed ConvNeXt-Tiny + VG-LayerCAM across all 6 models on Intel Natural Scenes."""
    print(f"\n{'#' * 68}\nPART 1: CONVNEXT-TINY (3 SEEDS) & VG-LAYERCAM ON INTEL SCENES\n{'#' * 68}")
    seeds = [0, 1, 2]
    all_models = [
        "resnet50",
        "efficientnet_b0",
        "convnext_tiny",
        "swin_tiny",
        "clip_zeroshot",
        "clip_linear",
    ]

    convnext_runs: List[Dict[str, Any]] = []
    vglayercam_by_model: Dict[str, List[Dict[str, float]]] = {m: [] for m in all_models}
    cam_sim_by_model: Dict[str, float] = {}
    seed0_preds: Dict[str, List[int]] = {}
    seed0_y_true: List[int] = []

    for seed in seeds:
        set_seed(seed)
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            dataset_name="intel", img_size=224, batch_size=32, seed=seed
        )
        num_classes = len(class_names)

        # Fast 4,800-image stratified training loader (800/class) for rapid high-accuracy fine-tuning
        tr_ds = train_loader.dataset
        sub_idx = get_stratified_indices(tr_ds.targets, max_samples=4800, seed=seed)
        fast_tr_ds = TransformSubset(tr_ds, sub_idx, transform=None, raw_targets=tr_ds.targets)
        fast_train_loader = DataLoader(
            fast_tr_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
        )

        # 1. Train & full benchmark ConvNeXt-Tiny across seeds 0, 1, 2
        print(f"\n--- ConvNeXt-Tiny (seed={seed}) ---")
        set_seed(seed)
        cn_model = build_model("convnext_tiny", num_classes=num_classes, pretrained=True)
        cn_model, _, cn_ckpt = train_model(
            model=cn_model,
            model_name="convnext_tiny",
            train_loader=fast_train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            epochs=5,
            lr=2e-4,
            checkpoint_dir=str(ckpt_dir),
            tag=f"intel_seed{seed}",
            resume=True,
        )
        eval_res = evaluate_model(cn_model, test_loader, device, class_names)
        stats = get_model_stats(cn_model, device, img_size=224)
        xai_faith = evaluate_xai_faithfulness(
            cn_model,
            "convnext_tiny",
            test_loader,
            device,
            methods=("gradcam", "gradcam++", "layercam", "vg_layercam"),
            max_samples=60,
            steps=20,
            seed=seed,
        )
        fgsm_res = evaluate_adversarial_robustness(
            cn_model, test_loader, device, attack_type="fgsm", max_samples=600, seed=seed
        )
        fgsm_pred_res = evaluate_adversarial_robustness(
            cn_model, test_loader, device, attack_type="fgsm_pred", max_samples=600, seed=seed
        )
        pgd_res = evaluate_adversarial_robustness(
            cn_model, test_loader, device, attack_type="pgd", max_samples=600, seed=seed
        )

        cn_record = {
            "seed": seed,
            "checkpoint": str(cn_ckpt),
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
        convnext_runs.append(cn_record)
        vglayercam_by_model["convnext_tiny"].append(xai_faith["vg_layercam"])

        if seed == 0:
            seed0_preds["convnext_tiny"] = eval_res["y_pred"].tolist()
            seed0_y_true = eval_res["y_true"].tolist()
            cam_sim_by_model["convnext_tiny"] = evaluate_adversarial_cam_similarity(
                cn_model, "convnext_tiny", test_loader, device, max_samples=48, seed=0
            )["cam_cosine_sim"]
            plot_confusion_matrix(
                eval_res["confusion_matrix"],
                class_names,
                title="Confusion Matrix (convnext_tiny)",
                save_path=str(out_dir / "cm_convnext_tiny.png"),
            )
            plot_xai_comparison_grid(
                cn_model,
                "convnext_tiny",
                test_loader,
                class_names,
                device,
                save_path=str(out_dir / "xai_grid_convnext_tiny.png"),
                num_samples=4,
            )

        # 2. Evaluate VG-LayerCAM across the other 5 models for this seed
        for m_name in all_models:
            if m_name == "convnext_tiny":
                continue
            if m_name == "clip_zeroshot" and vglayercam_by_model["clip_zeroshot"]:
                vglayercam_by_model["clip_zeroshot"].append(
                    vglayercam_by_model["clip_zeroshot"][0]
                )
                continue

            set_seed(seed)
            m_obj = build_model(
                m_name, num_classes=num_classes, pretrained=True, class_names=class_names
            )
            ep = 0 if m_name == "clip_zeroshot" else (3 if m_name == "clip_linear" else 3)
            lr = 1e-3 if m_name == "clip_linear" else 2e-4
            m_obj, _, _ = train_model(
                model=m_obj,
                model_name=m_name,
                train_loader=fast_train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                epochs=ep,
                lr=lr,
                checkpoint_dir=str(ckpt_dir),
                tag=f"intel_seed{seed}",
                resume=True,
            )
            vg_res = evaluate_xai_faithfulness(
                m_obj,
                m_name,
                test_loader,
                device,
                methods=("vg_layercam",),
                max_samples=60,
                steps=20,
                seed=seed,
            )["vg_layercam"]
            vglayercam_by_model[m_name].append(vg_res)
            print(f"[{m_name} | seed={seed}] VG-LayerCAM: {vg_res}")

            if seed == 0:
                ev0 = evaluate_model(m_obj, test_loader, device, class_names)
                seed0_preds[m_name] = ev0["y_pred"].tolist()
                cam_sim_by_model[m_name] = evaluate_adversarial_cam_similarity(
                    m_obj, m_name, test_loader, device, max_samples=48, seed=0
                )["cam_cosine_sim"]

    # Aggregate VG-LayerCAM mean/std across 3 seeds
    import numpy as np

    vg_summary = {}
    for m_name, runs in vglayercam_by_model.items():
        ins_arr = np.array([r["insertion_auc"] for r in runs])
        del_arr = np.array([r["deletion_auc"] for r in runs])
        dlt_arr = np.array([r["delta_auc"] for r in runs])
        vg_summary[m_name] = {
            "insertion_auc": {"mean": round(float(ins_arr.mean()), 4), "std": round(float(ins_arr.std()), 4)},
            "deletion_auc": {"mean": round(float(del_arr.mean()), 4), "std": round(float(del_arr.std()), 4)},
            "delta_auc": {"mean": round(float(dlt_arr.mean()), 4), "std": round(float(dlt_arr.std()), 4)},
            "clean_vs_pgd_cam_cosine_sim": cam_sim_by_model.get(m_name, 0.0),
        }

    payload = {
        "convnext_tiny_aggregated": aggregate_seed_metrics(convnext_runs),
        "convnext_tiny_per_seed": convnext_runs,
        "vg_layercam_3seed": vg_summary,
        "mcnemar_6models_intel": compute_pairwise_mcnemar(seed0_y_true, seed0_preds),
    }
    with open(res_dir / "stage3_intel_convnext_vglayercam.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _save_zips(res_dir, out_dir)


def run_sac_pgd_at_ablation(
    device: torch.device, ckpt_dir: Path, out_dir: Path, res_dir: Path
) -> None:
    """Run SAC-PGD-AT beta ablation study on ResNet-50, ConvNeXt-Tiny, and Swin-Tiny."""
    print(f"\n{'#' * 68}\nPART 2: SAC-PGD-AT ABLATION STUDY (beta in {{0.0, 0.05, 0.1, 0.2}})\n{'#' * 68}")
    set_seed(0)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_name="intel", img_size=224, batch_size=32, seed=0
    )
    num_classes = len(class_names)

    tr_ds = train_loader.dataset
    sub_idx = get_stratified_indices(tr_ds.targets, max_samples=3600, seed=0)
    fast_tr_ds = TransformSubset(tr_ds, sub_idx, transform=None, raw_targets=tr_ds.targets)
    fast_train_loader = DataLoader(
        fast_tr_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )

    sample_imgs, sample_lbls = next(iter(test_loader))
    sample_img = sample_imgs[0:1]
    sample_lbl = int(sample_lbls[0].item())

    ablation_results: Dict[str, Any] = {}
    # Full beta grid {0.0, 0.05, 0.1, 0.2} on Swin-Tiny and ConvNeXt-Tiny; {0.0, 0.1} on ResNet-50
    model_betas = {
        "resnet50": [0.0, 0.1],
        "convnext_tiny": [0.0, 0.05, 0.1, 0.2],
        "swin_tiny": [0.0, 0.05, 0.1, 0.2],
    }

    for model_name, betas in model_betas.items():
        ablation_results[model_name] = {}
        clean_ckpt = ckpt_dir / f"best_{model_name}_intel_seed0.pth"

        clean_model_ref = build_model(model_name, num_classes=num_classes, pretrained=True)
        if clean_ckpt.is_file():
            clean_model_ref.load_state_dict(
                torch.load(clean_ckpt, map_location=device, weights_only=False)["model_state_dict"]
            )
        clean_model_ref.to(device).eval()

        pgd_at_model_ref = None
        sac_01_model_ref = None

        for beta in betas:
            print(f"\n--- {model_name} | SAC-PGD-AT (beta={beta}) ---")
            set_seed(0)
            model = build_model(model_name, num_classes=num_classes, pretrained=True)
            if clean_ckpt.is_file():
                model.load_state_dict(
                    torch.load(clean_ckpt, map_location=device, weights_only=False)["model_state_dict"]
                )

            tag = f"sac_b{str(beta).replace('.', '')}"
            model, hist, ckpt_path = train_sac_pgd_adversarial(
                model=model,
                model_name=model_name,
                train_loader=fast_train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                epochs=3,
                lr=1e-4,
                epsilon=0.03,
                pgd_steps=2,
                clean_weight=0.5,
                beta_sac=beta,
                checkpoint_dir=str(ckpt_dir),
                tag=tag,
            )

            eval_res = evaluate_model(model, test_loader, device, class_names)
            pgd_res = evaluate_adversarial_robustness(
                model,
                test_loader,
                device,
                epsilons=(0.0, 0.01, 0.03, 0.05),
                attack_type="pgd",
                max_samples=600,
                seed=0,
            )
            xai_res = evaluate_xai_faithfulness(
                model,
                model_name,
                test_loader,
                device,
                methods=("gradcam", "gradcam++", "layercam", "vg_layercam"),
                max_samples=60,
                steps=20,
                seed=0,
            )
            cam_sim = evaluate_adversarial_cam_similarity(
                model, model_name, test_loader, device, epsilon=0.03, method="vg_layercam", max_samples=48, seed=0
            )

            ablation_results[model_name][str(beta)] = {
                "beta_sac": beta,
                "checkpoint": str(ckpt_path),
                "history": hist,
                "test_accuracy": round(eval_res["accuracy"], 4),
                "macro_f1": round(eval_res["macro_f1"], 4),
                "pgd_accuracy": pgd_res,
                "xai_faithfulness": xai_res,
                "adversarial_cam_similarity": cam_sim,
            }

            if beta == 0.0:
                pgd_at_model_ref = model
            elif beta == 0.1:
                sac_01_model_ref = model

        if pgd_at_model_ref is not None and sac_01_model_ref is not None:
            plot_sac_drift_comparison(
                clean_model_ref,
                pgd_at_model_ref,
                sac_01_model_ref,
                model_name,
                sample_img,
                sample_lbl,
                class_names,
                device,
                epsilon=0.03,
                save_path=str(out_dir / f"adv_drift_{model_name}_sac_pgd_at.png"),
            )

        with open(res_dir / "stage3_sac_ablation_intel.json", "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)
        _save_zips(res_dir, out_dir)


def run_sun397_convnext_vglayercam_sac(
    device: torch.device, ckpt_dir: Path, out_dir: Path, res_dir: Path
) -> None:
    """Run ConvNeXt-Tiny, VG-LayerCAM across all 6 models, and SAC-PGD-AT on SUN397-100."""
    print(f"\n{'#' * 68}\nPART 3: SUN397-100 CONVNEXT-TINY, VG-LAYERCAM & SAC-PGD-AT\n{'#' * 68}")
    set_seed(0)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_name="sun397_100", img_size=224, batch_size=32, seed=0, max_classes=100
    )
    num_classes = len(class_names)

    # 1. ConvNeXt-Tiny on SUN397-100
    cn_model = build_model("convnext_tiny", num_classes=num_classes, pretrained=True)
    cn_model, _, cn_ckpt = train_model(
        model=cn_model,
        model_name="convnext_tiny",
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        epochs=5,
        lr=2e-4,
        checkpoint_dir=str(ckpt_dir),
        tag="sun397_100_seed0",
        resume=True,
    )
    cn_eval = evaluate_model(cn_model, test_loader, device, class_names)
    cn_stats = get_model_stats(cn_model, device, img_size=224)
    cn_xai = evaluate_xai_faithfulness(
        cn_model,
        "convnext_tiny",
        test_loader,
        device,
        methods=("gradcam", "gradcam++", "layercam", "vg_layercam"),
        max_samples=200,
        steps=10,
        seed=0,
    )
    cn_fgsm = evaluate_adversarial_robustness(
        cn_model, test_loader, device, attack_type="fgsm", max_samples=500, seed=0
    )
    cn_pgd = evaluate_adversarial_robustness(
        cn_model, test_loader, device, attack_type="pgd", max_samples=500, seed=0
    )

    # 2. VG-LayerCAM on the other 5 models on SUN397-100
    other_models = ["resnet50", "efficientnet_b0", "swin_tiny", "clip_zeroshot", "clip_linear"]
    sun_vglayercam: Dict[str, Any] = {"convnext_tiny": cn_xai["vg_layercam"]}

    for m_name in other_models:
        set_seed(0)
        m_obj = build_model(
            m_name, num_classes=num_classes, pretrained=True, class_names=class_names
        )
        ep = 0 if m_name == "clip_zeroshot" else (5 if m_name == "clip_linear" else 4)
        lr = 1e-3 if m_name == "clip_linear" else 2e-4
        m_obj, _, _ = train_model(
            model=m_obj,
            model_name=m_name,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            epochs=ep,
            lr=lr,
            checkpoint_dir=str(ckpt_dir),
            tag="sun397_100_seed0",
            resume=True,
        )
        vg_m = evaluate_xai_faithfulness(
            m_obj,
            m_name,
            test_loader,
            device,
            methods=("vg_layercam",),
            max_samples=200,
            steps=10,
            seed=0,
        )["vg_layercam"]
        sun_vglayercam[m_name] = vg_m
        print(f"[SUN397-100 | {m_name}] VG-LayerCAM: {vg_m}")

    # 3. SAC-PGD-AT (beta=0.1) vs Standard PGD-AT (beta=0.0) on SUN397-100 for ConvNeXt-Tiny & Swin-Tiny
    tr_ds = train_loader.dataset
    sub_idx = get_stratified_indices(tr_ds.targets, max_samples=2500, seed=0)
    fast_sun_tr = DataLoader(
        TransformSubset(tr_ds, sub_idx, transform=None, raw_targets=tr_ds.targets),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    sun_sac_results: Dict[str, Any] = {}
    for m_name in ("convnext_tiny", "swin_tiny"):
        sun_sac_results[m_name] = {}
        base_ckpt = ckpt_dir / f"best_{m_name}_sun397_100_seed0.pth"
        for beta in (0.0, 0.1):
            set_seed(0)
            m_at = build_model(m_name, num_classes=num_classes, pretrained=True)
            if base_ckpt.is_file():
                m_at.load_state_dict(
                    torch.load(base_ckpt, map_location=device, weights_only=False)["model_state_dict"]
                )
            m_at, _, _ = train_sac_pgd_adversarial(
                model=m_at,
                model_name=m_name,
                train_loader=fast_sun_tr,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                epochs=2,
                lr=1e-4,
                epsilon=0.03,
                pgd_steps=2,
                beta_sac=beta,
                checkpoint_dir=str(ckpt_dir),
                tag=f"sun397_sac_b{str(beta).replace('.', '')}",
            )
            ev_at = evaluate_model(m_at, test_loader, device, class_names)
            pgd_at = evaluate_adversarial_robustness(
                m_at, test_loader, device, epsilons=(0.0, 0.01, 0.03), attack_type="pgd", max_samples=500, seed=0
            )
            xai_at = evaluate_xai_faithfulness(
                m_at, m_name, test_loader, device, methods=("gradcam", "vg_layercam"), max_samples=200, steps=10, seed=0
            )
            sim_at = evaluate_adversarial_cam_similarity(
                m_at, m_name, test_loader, device, epsilon=0.03, method="vg_layercam", max_samples=50, seed=0
            )
            sun_sac_results[m_name][str(beta)] = {
                "beta_sac": beta,
                "test_accuracy": round(ev_at["accuracy"], 4),
                "macro_f1": round(ev_at["macro_f1"], 4),
                "pgd_accuracy": pgd_at,
                "xai_faithfulness": xai_at,
                "adversarial_cam_similarity": sim_at,
            }

    sun_payload = {
        "convnext_tiny_sun397_100": {
            "checkpoint": str(cn_ckpt),
            "test_accuracy": round(cn_eval["accuracy"], 4),
            "macro_f1": round(cn_eval["macro_f1"], 4),
            "weighted_f1": round(cn_eval["weighted_f1"], 4),
            "stats": cn_stats,
            "xai_faithfulness": cn_xai,
            "fgsm_accuracy": cn_fgsm,
            "pgd_accuracy": cn_pgd,
        },
        "sun397_100_vg_layercam": sun_vglayercam,
        "sun397_100_sac_pgd_at": sun_sac_results,
    }
    with open(res_dir / "stage3_sun397_convnext_vglayercam_sac.json", "w", encoding="utf-8") as f:
        json.dump(sun_payload, f, indent=2)
    _save_zips(res_dir, out_dir)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("./checkpoints")
    out_dir = Path("./outputs")
    res_dir = Path("./results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    run_intel_convnext_and_vglayercam(device, ckpt_dir, out_dir, res_dir)
    run_sac_pgd_at_ablation(device, ckpt_dir, out_dir, res_dir)
    run_sun397_convnext_vglayercam_sac(device, ckpt_dir, out_dir, res_dir)
    print("\nALL STAGE 3 NEUROCOMPUTING EXPERIMENTS COMPLETE!")


if __name__ == "__main__":
    main()
