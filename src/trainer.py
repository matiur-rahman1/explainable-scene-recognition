"""Training, checkpointing (with save/resume), and evaluation pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Run a single training or validation epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = device.type == "cuda"

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc="Train" if is_train else "Val"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / max(1, total), correct / max(1, total)


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    epochs: int = 8,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
    tag: Optional[str] = None,
    resume: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]], Path]:
    """Fine-tune a model with AdamW, CosineAnnealingLR, and best-val checkpointing.

    Automatically handles zero-shot models (0 trainable parameters) and supports
    resuming from an existing checkpoint when resume=True.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_{model_name}_{tag}.pth" if tag else f"best_{model_name}.pth"
    best_path = ckpt_dir / ckpt_name

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params or epochs <= 0:
        # Zero-shot model: evaluate once on val_loader and save checkpoint
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        print(f"[{model_name}] Zero-Shot Val Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%")
        history = {
            "train_loss": [val_loss],
            "train_acc": [val_acc],
            "val_loss": [val_loss],
            "val_acc": [val_acc],
        }
        torch.save(
            {
                "model_name": model_name,
                "class_names": class_names,
                "val_acc": val_acc,
                "history": history,
                "model_state_dict": model.state_dict(),
            },
            best_path,
        )
        return model, history, best_path

    if resume and best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt and ckpt.get("completed_epochs", 0) >= epochs:
            print(f"[{model_name}] Resuming completed checkpoint from {best_path}")
            model.load_state_dict(ckpt["model_state_dict"])
            return model, ckpt.get("history", {}), best_path

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 5))
        history["train_acc"].append(round(tr_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": model_name,
                    "class_names": class_names,
                    "val_acc": best_val_acc,
                    "completed_epochs": epoch,
                    "history": history,
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Mark full completion in saved checkpoint for future resume
    checkpoint["completed_epochs"] = epochs
    checkpoint["history"] = history
    torch.save(checkpoint, best_path)
    return model, history, best_path


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluate a trained model on a test DataLoader and return detailed metrics."""
    model.eval()
    model.to(device)
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, leave=False, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.concatenate(all_probs, axis=0)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class_acc: Dict[str, float] = {}
    for idx, cls_name in enumerate(class_names):
        cls_total = int(cm[idx].sum())
        cls_corr = int(cm[idx, idx])
        per_class_acc[cls_name] = round(cls_corr / max(1, cls_total), 4)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "report": report,
    }
