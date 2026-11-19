"""Adversarial robustness evaluation (FGSM, PGD) and PGD Adversarial Training (PGD-AT)."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, build_stratified_eval_loader


def _norm_bounds(device: torch.device):
    """Return per-channel min/max bounds and std tensor in normalized image space."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper, std


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    use_pred_label: bool = False,
) -> torch.Tensor:
    """Generate Fast Gradient Sign Method (FGSM) adversarial examples.

    Args:
        model: Target classification model.
        images: Normalized input batch (B, 3, H, W).
        labels: Ground-truth labels (B,).
        epsilon: L-infinity perturbation budget in [0, 1] pixel scale.
        use_pred_label: If True, maximize loss w.r.t. model's predicted label
            to avoid the label-leaking effect on label-smoothed models (Kurakin et al., 2017).
    """
    if epsilon == 0.0:
        return images.detach().clone()

    model.eval()
    device = images.device
    lower, upper, std = _norm_bounds(device)
    eps_scaled = epsilon / std

    adv = images.detach().clone().requires_grad_(True)
    logits = model(adv)
    target = logits.argmax(dim=1).detach() if use_pred_label else labels
    loss = F.cross_entropy(logits, target)
    model.zero_grad(set_to_none=True)
    loss.backward()

    with torch.no_grad():
        adv = adv + eps_scaled * adv.grad.sign()
        adv = torch.max(torch.min(adv, upper), lower)
    return adv.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: Optional[float] = None,
    num_steps: int = 10,
) -> torch.Tensor:
    """Generate Projected Gradient Descent (PGD-Linf) adversarial examples.

    Automatically scales step size alpha = 2.5 * epsilon / num_steps when alpha is None
    so that the full L-infinity ball boundary is reachable for any epsilon.
    """
    if epsilon == 0.0:
        return images.detach().clone()

    model.eval()
    device = images.device
    lower, upper, std = _norm_bounds(device)
    step_val = alpha if alpha is not None else (2.5 * epsilon / max(1, num_steps))
    eps_scaled = epsilon / std
    alpha_scaled = step_val / std

    orig = images.detach()
    delta = torch.empty_like(orig).uniform_(-1.0, 1.0) * eps_scaled
    adv = torch.max(torch.min(orig + delta, upper), lower).detach()

    for _ in range(num_steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            adv = adv + alpha_scaled * adv.grad.sign()
            eta = torch.clamp(adv - orig, min=-eps_scaled, max=eps_scaled)
            adv = torch.max(torch.min(orig + eta, upper), lower).detach()

    return adv


def evaluate_adversarial_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilons: Iterable[float] = (0.0, 0.01, 0.03, 0.05),
    attack_type: str = "fgsm",
    max_samples: int = 600,
    seed: int = 42,
) -> Dict[float, float]:
    """Measure classification accuracy under FGSM or PGD on a class-stratified test subset."""
    model.eval()
    model.to(device)
    attack_key = attack_type.lower()
    results: Dict[float, float] = {}

    strat_loader = build_stratified_eval_loader(
        loader, max_samples=max_samples, batch_size=loader.batch_size or 32, seed=seed
    )

    for eps in epsilons:
        correct = 0
        total = 0
        for images, labels in strat_loader:
            images = images.to(device)
            labels = labels.to(device)

            if attack_key == "fgsm":
                adv_images = fgsm_attack(model, images, labels, epsilon=eps, use_pred_label=False)
            elif attack_key == "fgsm_pred":
                adv_images = fgsm_attack(model, images, labels, epsilon=eps, use_pred_label=True)
            elif attack_key == "pgd":
                adv_images = pgd_attack(model, images, labels, epsilon=eps, num_steps=10)
            else:
                raise ValueError(
                    f"Unsupported attack '{attack_type}'. Use 'fgsm', 'fgsm_pred', or 'pgd'."
                )

            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        results[float(eps)] = round(correct / max(1, total), 4)

    return results


def train_pgd_adversarial(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    epsilon: float = 0.03,
    pgd_steps: int = 5,
    clean_weight: float = 0.5,
    checkpoint_dir: str = "./checkpoints",
) -> Tuple[nn.Module, Dict[str, List[float]], Path]:
    """Fine-tune a model with mixed Clean + PGD Adversarial Training (PGD-AT, eps=0.03)."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_{model_name}_pgd_at.pth"

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_clean_acc": [],
        "val_pgd_acc": [],
    }
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for images, labels in tqdm(train_loader, leave=False, desc=f"PGD-AT Ep {epoch}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Generate inner-maximization PGD adversarial examples
            adv_images = pgd_attack(
                model, images, labels, epsilon=epsilon, num_steps=pgd_steps
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits_clean = model(images)
            logits_adv = model(adv_images)
            loss = clean_weight * criterion(logits_clean, labels) + (
                1.0 - clean_weight
            ) * criterion(logits_adv, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        scheduler.step()
        tr_loss = total_loss / max(1, total)

        # Evaluate clean and PGD accuracy on a stratified validation subset
        val_clean = evaluate_adversarial_robustness(
            model, val_loader, device, epsilons=(0.0,), attack_type="pgd", max_samples=300
        )[0.0]
        val_pgd = evaluate_adversarial_robustness(
            model, val_loader, device, epsilons=(epsilon,), attack_type="pgd", max_samples=300
        )[float(epsilon)]

        history["train_loss"].append(round(tr_loss, 4))
        history["val_clean_acc"].append(val_clean)
        history["val_pgd_acc"].append(val_pgd)

        print(
            f"[{model_name}+PGD-AT] Epoch {epoch:02d}/{epochs:02d} | "
            f"Loss: {tr_loss:.4f} | Val Clean: {val_clean*100:.2f}% | "
            f"Val PGD(eps={epsilon}): {val_pgd*100:.2f}%"
        )

        combined_score = 0.5 * val_clean + 0.5 * val_pgd
        if combined_score > best_score:
            best_score = combined_score
            torch.save(
                {
                    "model_name": f"{model_name}_pgd_at",
                    "class_names": class_names,
                    "val_clean_acc": val_clean,
                    "val_pgd_acc": val_pgd,
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history, best_path
