"""Adversarial robustness evaluation using FGSM and PGD attacks (Proposal Phase 4)."""

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import IMAGENET_MEAN, IMAGENET_STD


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
) -> torch.Tensor:
    """Generate Fast Gradient Sign Method (FGSM) adversarial examples."""
    if epsilon == 0.0:
        return images.detach().clone()

    model.eval()
    device = images.device
    lower, upper, std = _norm_bounds(device)
    eps_scaled = epsilon / std

    adv = images.detach().clone().requires_grad_(True)
    logits = model(adv)
    loss = F.cross_entropy(logits, labels)
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
    alpha: float = 0.005,
    num_steps: int = 7,
) -> torch.Tensor:
    """Generate Projected Gradient Descent (PGD-Linf) adversarial examples."""
    if epsilon == 0.0:
        return images.detach().clone()

    model.eval()
    device = images.device
    lower, upper, std = _norm_bounds(device)
    eps_scaled = epsilon / std
    alpha_scaled = alpha / std

    orig = images.detach()
    # Random start within L-inf epsilon ball
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
    max_batches: int = 15,
) -> Dict[float, float]:
    """Measure classification accuracy under FGSM or PGD attacks across epsilons."""
    model.eval()
    model.to(device)
    attack_key = attack_type.lower()
    results: Dict[float, float] = {}

    for eps in epsilons:
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            if attack_key == "fgsm":
                adv_images = fgsm_attack(model, images, labels, epsilon=eps)
            elif attack_key == "pgd":
                adv_images = pgd_attack(model, images, labels, epsilon=eps)
            else:
                raise ValueError(f"Unsupported attack '{attack_type}'. Use 'fgsm' or 'pgd'.")

            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        results[float(eps)] = round(correct / max(1, total), 4)

    return results
