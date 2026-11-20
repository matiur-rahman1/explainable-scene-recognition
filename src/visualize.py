"""Publication-quality visualization helpers for training, XAI, and robustness analysis."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.adversarial import fgsm_attack
from src.dataset import denormalize_image
from src.xai import generate_cam_explanation


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
) -> None:
    """Plot validation accuracy and loss curves across all benchmarked models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, hist in histories.items():
        epochs = range(1, len(hist["val_acc"]) + 1)
        axes[0].plot(epochs, [a * 100 for a in hist["val_acc"]], marker="o", label=name)
        axes[1].plot(epochs, hist["val_loss"], marker="s", label=name)

    axes[0].set_title("Validation Accuracy Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Loss Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot a normalized/annotated confusion matrix."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_xai_comparison_grid(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    title_suffix: str = "",
) -> None:
    """Visualize Original | Grad-CAM | Grad-CAM++ | LayerCAM | VG-LayerCAM across diverse classes."""
    model.eval()
    model.to(device)
    methods = ["gradcam", "gradcam++", "layercam", "vg_layercam"]
    titles = ["Original", "Grad-CAM", "Grad-CAM++", "LayerCAM", "VG-LayerCAM (Ours)"]

    selected = []
    seen_classes = set()
    for images, labels in loader:
        for i in range(images.size(0)):
            lbl = int(labels[i].item())
            if lbl not in seen_classes:
                selected.append((images[i : i + 1], lbl))
                seen_classes.add(lbl)
                if len(selected) >= num_samples:
                    break
        if len(selected) >= num_samples:
            break

    rows = len(selected)
    fig, axes = plt.subplots(rows, 5, figsize=(17, 3.3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (img_t, true_idx) in enumerate(selected):
        x = img_t.to(device)
        rgb = denormalize_image(x[0])

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(f"True: {class_names[true_idx]}", fontsize=10, fontweight="bold")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[0, 0].set_title(titles[0], fontsize=11, fontweight="bold")

        for c_idx, method in enumerate(methods, start=1):
            _, overlay, pred_idx, pred_prob = generate_cam_explanation(
                model, model_name, x, rgb, method=method
            )
            axes[r, c_idx].imshow(overlay)
            axes[r, c_idx].set_xticks([])
            axes[r, c_idx].set_yticks([])
            if r == 0:
                axes[0, c_idx].set_title(titles[c_idx], fontsize=11, fontweight="bold")
            axes[r, c_idx].set_xlabel(
                f"Pred: {class_names[pred_idx]} ({pred_prob * 100:.1f}%)", fontsize=9
            )

    disp_title = f"Gradient-Based Visual Explanations ({model_name}{title_suffix})"
    plt.suptitle(disp_title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial_robustness(
    robustness_results: Dict[str, Dict[float, float]],
    attack_name: str = "FGSM",
    save_path: Optional[str] = None,
) -> None:
    """Plot accuracy degradation under adversarial perturbation across models."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model_name, eps_map in robustness_results.items():
        epsilons = sorted(eps_map.keys())
        accs = [eps_map[e] * 100 for e in epsilons]
        ax.plot(epsilons, accs, marker="o", linewidth=2, label=model_name)

    ax.set_title(
        f"CNN vs. Transformer Robustness under {attack_name} Attack",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Perturbation Budget (Epsilon)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial_attention_drift(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    label_idx: int,
    class_names: List[str],
    device: torch.device,
    epsilon: float = 0.03,
    save_path: Optional[str] = None,
) -> None:
    """Compare clean vs. adversarial predictions and Grad-CAM attention maps."""
    x = image_tensor.to(device)
    y = torch.tensor([label_idx], device=device)
    adv_x = fgsm_attack(model, x, y, epsilon=epsilon)

    clean_rgb = denormalize_image(x[0])
    adv_rgb = denormalize_image(adv_x[0])

    _, clean_cam, clean_pred, clean_prob = generate_cam_explanation(
        model, model_name, x, clean_rgb, method="gradcam"
    )
    _, adv_cam, adv_pred, adv_prob = generate_cam_explanation(
        model, model_name, adv_x, adv_rgb, method="gradcam"
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    panels = [
        (clean_rgb, f"Clean\nTrue: {class_names[label_idx]}"),
        (clean_cam, f"Clean Grad-CAM\nPred: {class_names[clean_pred]} ({clean_prob*100:.1f}%)"),
        (adv_rgb, f"FGSM (eps={epsilon})"),
        (adv_cam, f"Adversarial Grad-CAM\nPred: {class_names[adv_pred]} ({adv_prob*100:.1f}%)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Adversarial Attention Drift Analysis ({model_name})",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_failure_cases(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    save_path: Optional[str] = None,
    max_cases: int = 4,
) -> None:
    """Collect misclassified test examples and visualize their Grad-CAM failure modes."""
    model.eval()
    model.to(device)
    failures = []

    for images, labels in loader:
        images_dev = images.to(device)
        with torch.no_grad():
            preds = model(images_dev).argmax(dim=1).cpu()
        for i in range(images.size(0)):
            if preds[i].item() != labels[i].item():
                failures.append((images[i : i + 1], int(labels[i].item())))
                if len(failures) >= max_cases:
                    break
        if len(failures) >= max_cases:
            break

    if not failures:
        print(f"No misclassified samples found for {model_name}!")
        return

    rows = len(failures)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (img_t, true_idx) in enumerate(failures):
        x = img_t.to(device)
        rgb = denormalize_image(x[0])
        _, overlay, pred_idx, pred_prob = generate_cam_explanation(
            model, model_name, x, rgb, method="gradcam"
        )
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"True: {class_names[true_idx]}", fontsize=10, fontweight="bold")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(overlay)
        axes[r, 1].set_title(
            f"Misclassified: {class_names[pred_idx]} ({pred_prob*100:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color="crimson",
        )
        axes[r, 1].axis("off")

    plt.suptitle(
        f"Failure Case Visual Explanation ({model_name})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
