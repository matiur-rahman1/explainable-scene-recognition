"""Adversarial attacks (FGSM, PGD), PGD-AT, and Spatial Attribution-Consistent Adversarial Training (SAC-PGD-AT)."""

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, build_stratified_eval_loader
from src.models import get_target_layers_and_reshape
from src.xai import generate_cam_explanation


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
    """Generate Fast Gradient Sign Method (FGSM) adversarial examples."""
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
    """Generate Projected Gradient Descent (PGD-Linf) adversarial examples."""
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


def evaluate_adversarial_cam_similarity(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.03,
    method: str = "vg_layercam",
    max_samples: int = 60,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute Cosine Similarity between clean and PGD adversarial CAM heatmaps: Sim(S^c(x), S^c(x_PGD))."""
    model.eval()
    model.to(device)
    strat_loader = build_stratified_eval_loader(
        loader, max_samples=max_samples, batch_size=1, seed=seed
    )

    cos_sims: List[float] = []
    l1_drifts: List[float] = []
    processed = 0

    for images, labels in strat_loader:
        if processed >= max_samples:
            break
        x = images.to(device)
        y = labels.to(device)
        adv_x = pgd_attack(model, x, y, epsilon=epsilon, num_steps=5)

        true_cls = int(y[0].item())
        dummy_rgb = np.zeros((x.shape[2], x.shape[3], 3), dtype=np.float32)

        cam_clean, _, _, _ = generate_cam_explanation(
            model, model_name, x, dummy_rgb, method=method, target_class=true_cls
        )
        cam_adv, _, _, _ = generate_cam_explanation(
            model, model_name, adv_x, dummy_rgb, method=method, target_class=true_cls
        )

        v1 = cam_clean.reshape(-1).astype(np.float64)
        v2 = cam_adv.reshape(-1).astype(np.float64)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-7 and norm2 > 1e-7:
            cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
        else:
            cos_sim = 0.0
        l1_drift = float(np.mean(np.abs(v1 - v2)))

        cos_sims.append(cos_sim)
        l1_drifts.append(l1_drift)
        processed += 1

    return {
        "cam_cosine_sim": round(float(np.mean(cos_sims)), 4),
        "cam_cosine_std": round(float(np.std(cos_sims)), 4),
        "cam_l1_drift": round(float(np.mean(l1_drifts)), 4),
    }


def _compute_spatial_energy_map(
    activation: torch.Tensor,
    reshape_tf: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute L2-normalized spatial energy map E_bar(x) of shape (B, u*v) from target layer activation."""
    act = reshape_tf(activation) if reshape_tf is not None else activation
    # act has shape (B, C, u, v); channel L2 norm at each spatial location (i, j)
    spatial_l2 = torch.sqrt(torch.sum(act ** 2, dim=1) + 1e-8)  # (B, u, v)
    flat = spatial_l2.reshape(act.size(0), -1)  # (B, u*v)
    return F.normalize(flat, p=2, dim=1)


def train_sac_pgd_adversarial(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    epsilon: float = 0.03,
    pgd_steps: int = 2,
    clean_weight: float = 0.5,
    beta_sac: float = 0.1,
    checkpoint_dir: str = "./checkpoints",
    tag: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], Path]:
    """Proposed Spatial Attribution-Consistent Adversarial Training (SAC-PGD-AT).

    Minimizes:
        L_{SAC-AT} = lambda * L_CE(f(x), y) + (1 - lambda) * L_CE(f(x_PGD), y)
                     + beta_sac * || E_bar(x) - E_bar(x_PGD) ||_F^2
    where E_bar(x) is the L2-normalized spatial activation energy map at the CAM target layer.
    When beta_sac == 0.0, reduces to standard mixed PGD-AT.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = tag if tag else f"sac_b{str(beta_sac).replace('.', '')}"
    best_path = ckpt_dir / f"best_{model_name}_{suffix}.pth"

    model = model.to(device)
    target_layers, reshape_tf = get_target_layers_and_reshape(model, model_name)
    target_layer = target_layers[0]

    captured: Dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inp, out):
        captured["act"] = out

    hook_handle = target_layer.register_forward_hook(_forward_hook)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "sac_loss": [],
        "val_clean_acc": [],
        "val_pgd_acc": [],
    }
    best_score = -1.0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_sac = 0.0
            total = 0

            for images, labels in tqdm(
                train_loader, leave=False, desc=f"SAC-AT(b={beta_sac}) Ep {epoch}"
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                adv_images = pgd_attack(
                    model, images, labels, epsilon=epsilon, num_steps=pgd_steps
                )

                model.train()
                optimizer.zero_grad(set_to_none=True)

                logits_clean = model(images)
                e_clean = _compute_spatial_energy_map(captured["act"], reshape_tf)

                logits_adv = model(adv_images)
                e_adv = _compute_spatial_energy_map(captured["act"], reshape_tf)

                cls_loss = clean_weight * criterion(logits_clean, labels) + (
                    1.0 - clean_weight
                ) * criterion(logits_adv, labels)

                # Squared Frobenius distance per sample between normalized spatial energy maps
                sac_penalty = torch.mean(torch.sum((e_clean - e_adv) ** 2, dim=1))
                loss = cls_loss + beta_sac * sac_penalty

                loss.backward()
                optimizer.step()

                bs = images.size(0)
                total_loss += loss.item() * bs
                total_sac += sac_penalty.item() * bs
                total += bs

            scheduler.step()
            tr_loss = total_loss / max(1, total)
            tr_sac = total_sac / max(1, total)

            val_clean = evaluate_adversarial_robustness(
                model, val_loader, device, epsilons=(0.0,), attack_type="pgd", max_samples=300
            )[0.0]
            val_pgd = evaluate_adversarial_robustness(
                model, val_loader, device, epsilons=(epsilon,), attack_type="pgd", max_samples=300
            )[float(epsilon)]

            history["train_loss"].append(round(tr_loss, 4))
            history["sac_loss"].append(round(tr_sac, 4))
            history["val_clean_acc"].append(val_clean)
            history["val_pgd_acc"].append(val_pgd)

            print(
                f"[{model_name}+SAC-AT(b={beta_sac})] Ep {epoch:02d}/{epochs:02d} | "
                f"Loss: {tr_loss:.4f} (SAC: {tr_sac:.4f}) | "
                f"Val Clean: {val_clean*100:.2f}% | Val PGD({epsilon}): {val_pgd*100:.2f}%"
            )

            combined_score = 0.5 * val_clean + 0.5 * val_pgd
            if combined_score > best_score:
                best_score = combined_score
                torch.save(
                    {
                        "model_name": f"{model_name}_{suffix}",
                        "beta_sac": beta_sac,
                        "class_names": class_names,
                        "val_clean_acc": val_clean,
                        "val_pgd_acc": val_pgd,
                        "model_state_dict": model.state_dict(),
                    },
                    best_path,
                )
    finally:
        hook_handle.remove()

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history, best_path


def train_pgd_adversarial(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    epsilon: float = 0.03,
    pgd_steps: int = 2,
    clean_weight: float = 0.5,
    checkpoint_dir: str = "./checkpoints",
) -> Tuple[nn.Module, Dict[str, List[float]], Path]:
    """Standard mixed PGD-AT (special case of SAC-PGD-AT with beta_sac=0.0)."""
    return train_sac_pgd_adversarial(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        epochs=epochs,
        lr=lr,
        epsilon=epsilon,
        pgd_steps=pgd_steps,
        clean_weight=clean_weight,
        beta_sac=0.0,
        checkpoint_dir=checkpoint_dir,
        tag="pgd_at",
    )
