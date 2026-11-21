#!/usr/bin/env python3
"""
Session S1-train-a: Unified Full-Dataset Intel Natural Scenes Training & Clean Evaluation
========================================================================================
Models (3): resnet50, efficientnet_b0, convnext_tiny
Seeds  (3): 0, 1, 2
Total Units: 9 units (1 unit = 1 (dataset='intel', model, seed, epochs=6))

Protocol:
- Full Intel training split: 11,932 train images / 2,102 val images (85/15 stratified)
- Full Intel test split:     3,000 test images (evaluated in strict FP32, no AMP)
- Training: 6 epochs, batch_size=32, AdamW(lr=2e-4, weight_decay=1e-4),
  CosineAnnealingLR(T_max=6), CrossEntropyLoss(label_smoothing=0.05), AMP enabled for training only.
- Saves after EVERY unit:
  1. results/s1_train_intel_<model>_seed<seed>.json (includes history, test_metrics, y_true, y_pred, env metadata)
  2. checkpoints/best_<model>_intel_seed<seed>.pth (compact FP16 state_dict for S1a/S2a/S4/S7 reuse)
  3. results/s1_train_a_summary.json (aggregated 3-seed mean ± std)
  4. /kaggle/working/s1_train_a_results.zip (auto-refreshed after each unit)
"""

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import timm

# ---------------------------------------------------------------------------
# Constants & Registry
# ---------------------------------------------------------------------------
SESSION_ID = "S1-train-a"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TIMM_REGISTRY: Dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

DEFAULT_MODELS = ["resnet50", "efficientnet_b0", "convnext_tiny"]
DEFAULT_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Environment & Reproducibility Helpers
# ---------------------------------------------------------------------------
def detect_env_dirs(
    custom_out_root: Optional[str] = None,
) -> Tuple[str, Path, Path, Path, Path]:
    """Auto-detect Kaggle, Colab (with Drive mount), or Local environment."""
    if custom_out_root:
        root = Path(custom_out_root)
        env_name = "custom"
    elif Path("/kaggle/working").is_dir():
        root = Path("/kaggle/working")
        env_name = "kaggle"
    elif Path("/content").is_dir():
        env_name = "colab"
        drive_root = Path("/content/drive/MyDrive/scene_results")
        if not Path("/content/drive/MyDrive").is_dir():
            try:
                from google.colab import drive  # type: ignore

                drive.mount("/content/drive", force_remount=False)
            except Exception as e:
                print(f"[WARN] Could not auto-mount Colab Drive ({e}); falling back to /content/scene_results")
                drive_root = Path("/content/scene_results")
        root = drive_root
    else:
        root = Path.cwd()
        env_name = "local"

    res_dir = root / "results"
    ckpt_dir = root / "checkpoints"
    cache_dir = root / "pretrained_cache"
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Point HuggingFace / PyTorch / timm cache to persistent or input cache if available
    kaggle_cache_input = Path("/kaggle/input/scene-pretrained-cache")
    if kaggle_cache_input.is_dir():
        os.environ.setdefault("HF_HOME", str(kaggle_cache_input / "hf"))
        os.environ.setdefault("TORCH_HOME", str(kaggle_cache_input / "torch"))
        os.environ.setdefault("TIMM_CACHE_DIR", str(kaggle_cache_input / "timm"))
    else:
        os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
        os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
        os.environ.setdefault("TIMM_CACHE_DIR", str(cache_dir / "timm"))

    return env_name, root, res_dir, ckpt_dir, cache_dir


def sync_existing_units_from_kaggle_input(res_dir: Path, ckpt_dir: Path) -> None:
    """If previous session outputs are attached under /kaggle/input/*, copy unit JSONs into res_dir."""
    k_in = Path("/kaggle/input")
    if not k_in.is_dir():
        return
    for ds_dir in k_in.iterdir():
        if not ds_dir.is_dir():
            continue
        for cand_res in [ds_dir / "results", ds_dir]:
            if cand_res.is_dir():
                for jf in cand_res.glob("s1_train_intel_*.json"):
                    dst = res_dir / jf.name
                    if not dst.is_file():
                        shutil.copy2(jf, dst)
        for cand_ckpt in [ds_dir / "checkpoints", ds_dir]:
            if cand_ckpt.is_dir():
                for cf in cand_ckpt.glob("best_*_intel_seed*.pth"):
                    dst = ckpt_dir / cf.name
                    if not dst.is_file():
                        try:
                            os.symlink(cf.resolve(), dst)
                        except Exception:
                            shutil.copy2(cf, dst)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "standalone_kaggle_notebook"


def get_env_metadata(env_name: str) -> Dict[str, Any]:
    import torchvision

    gpu_name = "cpu"
    gpu_count = 0
    vram_gb = 0.0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

    return {
        "session_id": SESSION_ID,
        "environment": env_name,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "numpy_version": np.__version__,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "gpu_vram_gb": vram_gb,
    }


# ---------------------------------------------------------------------------
# Dataset Loading (Intel Natural Scenes)
# ---------------------------------------------------------------------------
class TransformSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform=None,
        raw_targets: Optional[Sequence[int]] = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        base_targets = raw_targets if raw_targets is not None else getattr(dataset, "targets", None)
        self.targets = [int(base_targets[i]) for i in self.indices] if base_targets is not None else []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return {"train": train_tf, "eval": eval_tf}


def find_intel_dirs(data_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """Locate Intel Natural Scenes seg_train and seg_test folders on Kaggle, local, or via kagglehub."""
    search_roots: List[Path] = []
    if data_dir:
        search_roots.append(Path(data_dir))

    # Common Kaggle input paths
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for p in sorted(kaggle_input.iterdir()):
            if p.is_dir():
                search_roots.append(p)

    search_roots.append(Path("./data/intel"))
    search_roots.append(Path("./data"))

    for root in search_roots:
        if not root.is_dir():
            continue
        candidates = [
            (root / "seg_train" / "seg_train", root / "seg_test" / "seg_test"),
            (root / "seg_train", root / "seg_test"),
        ]
        for tr, te in candidates:
            if tr.is_dir() and te.is_dir():
                subdirs = [d for d in tr.iterdir() if d.is_dir() and d.name != "seg_train"]
                if len(subdirs) == 6:
                    return tr, te

    # Fallback: auto-download with kagglehub if internet is enabled
    print("[INFO] Intel dataset not found in /kaggle/input; downloading via kagglehub...")
    import kagglehub  # type: ignore

    dl_root = Path(kagglehub.dataset_download("puneet6060/intel-image-classification"))
    for tr, te in [
        (dl_root / "seg_train" / "seg_train", dl_root / "seg_test" / "seg_test"),
        (dl_root / "seg_train", dl_root / "seg_test"),
    ]:
        if tr.is_dir() and te.is_dir():
            subdirs = [d for d in tr.iterdir() if d.is_dir() and d.name != "seg_train"]
            if len(subdirs) == 6:
                return tr, te
    raise FileNotFoundError(f"Could not find seg_train / seg_test under {dl_root}")


def stratified_split_indices(
    targets: Sequence[int], val_ratio: float = 0.15, seed: int = 0
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    train_idx, val_idx = [], []
    for cls_id in np.unique(targets_arr):
        cls_indices = np.where(targets_arr == cls_id)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_ratio))
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def create_intel_loaders(
    data_dir: Optional[str],
    img_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    tfs = get_transforms(img_size)
    train_dir, test_dir = find_intel_dirs(data_dir)
    base_train = datasets.ImageFolder(root=str(train_dir))
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=tfs["eval"])
    class_names = base_train.classes

    train_idx, val_idx = stratified_split_indices(
        base_train.targets, val_ratio=val_ratio, seed=seed
    )
    train_ds = TransformSubset(
        base_train, train_idx, transform=tfs["train"], raw_targets=base_train.targets
    )
    val_ds = TransformSubset(
        base_train, val_idx, transform=tfs["eval"], raw_targets=base_train.targets
    )

    pin = torch.cuda.is_available()
    pw = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    return train_loader, val_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# Model Building, Training (AMP), and Strict FP32 Evaluation
# ---------------------------------------------------------------------------
def build_timm_backbone(
    model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2
) -> nn.Module:
    key = model_name.lower()
    if key not in TIMM_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}' for S1-train-a.")
    return timm.create_model(
        TIMM_REGISTRY[key],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


@torch.no_grad()
def evaluate_fp32(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate model in strict FP32 (no AMP) and return loss, accuracy, F1, CM, y_true, y_pred."""
    model.eval()
    model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for b_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        # Strict FP32 forward pass (autocast disabled)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            logits = model(images)
            loss = criterion(logits, labels)

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class_acc: Dict[str, float] = {}
    for idx, cls_name in enumerate(class_names):
        cls_tot = int(cm[idx].sum())
        cls_cor = int(cm[idx, idx])
        per_class_acc[cls_name] = round(cls_cor / max(1, cls_tot), 4)

    return {
        "loss": round(total_loss / max(1, total), 5),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def measure_fp32_stats(
    model: nn.Module, device: torch.device, img_size: int = 224
) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=False):
            for _ in range(5):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = ((time.perf_counter() - t0) / 30.0) * 1000.0

    return {
        "total_params_m": round(total_params / 1e6, 2),
        "trainable_params_m": round(trainable_params / 1e6, 2),
        "latency_ms": round(latency_ms, 2),
    }


def train_and_evaluate_unit(
    model_name: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    ckpt_path: Path,
    dry_run: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Train one (model_name, seed) unit with AMP and evaluate in strict FP32."""
    set_seed(seed)
    num_classes = len(class_names)
    model = build_timm_backbone(model_name, num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 1,
        "best_val_acc": -1.0,
    }
    best_fp16_state: Dict[str, torch.Tensor] = {}

    run_epochs = 1 if dry_run else epochs
    max_batches = 2 if dry_run else None

    for ep in range(1, run_epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_sum = 0.0
        tr_corr = 0
        tr_tot = 0

        pbar = tqdm(
            train_loader,
            leave=False,
            desc=f"[{model_name}|s={seed}] Ep {ep:02d}/{run_epochs:02d}",
            disable=dry_run,
        )
        for b_idx, (images, labels) in enumerate(pbar):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            tr_loss_sum += float(loss.item()) * bs
            tr_corr += int((logits.argmax(dim=1) == labels).sum().item())
            tr_tot += bs

        scheduler.step()
        tr_loss = round(tr_loss_sum / max(1, tr_tot), 5)
        tr_acc = round(tr_corr / max(1, tr_tot), 5)

        val_res = evaluate_fp32(
            model, val_loader, device, class_names, criterion=criterion, max_batches=max_batches
        )
        val_loss = val_res["loss"]
        val_acc = val_res["accuracy"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = ep
            # Snapshot best weights in compact FP16 CPU tensors
            best_fp16_state = {
                k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
                for k, v in model.state_dict().items()
            }

        ep_sec = time.perf_counter() - ep_t0
        print(
            f"  [{model_name} | seed={seed}] Ep {ep:02d}/{run_epochs:02d} ({ep_sec:.1f}s) | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%"
        )

    # Restore best validation weights into FP32 model before full test evaluation
    restored_state = {
        k: (v.float() if v.is_floating_point() else v) for k, v in best_fp16_state.items()
    }
    model.load_state_dict(restored_state)

    test_res = evaluate_fp32(
        model, test_loader, device, class_names, criterion=criterion, max_batches=max_batches
    )
    stats = measure_fp32_stats(model, device, img_size=224)

    ckpt_size_mb = 0.0
    if not dry_run:
        ckpt_payload = {
            "session_id": SESSION_ID,
            "dataset": "intel",
            "model_name": model_name,
            "timm_id": TIMM_REGISTRY[model_name],
            "seed": seed,
            "num_classes": num_classes,
            "class_names": class_names,
            "best_epoch": history["best_epoch"],
            "best_val_acc": history["best_val_acc"],
            "test_accuracy": test_res["accuracy"],
            "macro_f1": test_res["macro_f1"],
            "fp16_state_dict": True,
            "model_state_dict": best_fp16_state,
        }
        torch.save(ckpt_payload, ckpt_path)
        ckpt_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

    return {
        "history": history,
        "test_metrics": test_res,
        "stats": stats,
        "checkpoint_size_mb": ckpt_size_mb,
    }, ep_sec


# ---------------------------------------------------------------------------
# Pretrained Weight Caching Helper (Rule f)
# ---------------------------------------------------------------------------
def cache_all_pretrained_weights(cache_dir: Path) -> None:
    """Download and cache all 6 architectures' pretrained weights so future sessions can run offline."""
    print(f"\n[CACHE] Downloading pretrained weights into {cache_dir} ...")
    for m_key, timm_name in TIMM_REGISTRY.items():
        t0 = time.perf_counter()
        _ = timm.create_model(timm_name, pretrained=True, num_classes=6)
        print(f"  Cached timm model '{m_key}' ({timm_name}) in {time.perf_counter()-t0:.1f}s")
    try:
        import open_clip  # type: ignore

        t0 = time.perf_counter()
        _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        print(f"  Cached open_clip 'ViT-B-16' ('openai') in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"  [INFO] open_clip not cached in S1-train-a ({e}); will be cached in S1-train-b.")


# ---------------------------------------------------------------------------
# Aggregation & Zip Sync Helpers
# ---------------------------------------------------------------------------
def update_summary_and_zips(
    root: Path, res_dir: Path, ckpt_dir: Path, models: List[str], seeds: List[int]
) -> None:
    """Aggregate completed unit JSONs into s1_train_a_summary.json and refresh download zips."""
    by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    completed_units: List[str] = []

    for m in models:
        for s in seeds:
            u_path = res_dir / f"s1_train_intel_{m}_seed{s}.json"
            if u_path.is_file():
                with open(u_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                by_model[m].append(data)
                completed_units.append(data["unit_id"])

    agg_summary: Dict[str, Any] = {}
    for m, runs in by_model.items():
        if not runs:
            continue
        accs = np.array([r["test_metrics"]["accuracy"] for r in runs], dtype=np.float64)
        mf1s = np.array([r["test_metrics"]["macro_f1"] for r in runs], dtype=np.float64)
        wf1s = np.array([r["test_metrics"]["weighted_f1"] for r in runs], dtype=np.float64)
        lats = np.array([r["stats"]["latency_ms"] for r in runs], dtype=np.float64)
        agg_summary[m] = {
            "num_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "test_accuracy": {"mean": round(float(accs.mean()), 4), "std": round(float(accs.std(ddof=0)), 4)},
            "macro_f1": {"mean": round(float(mf1s.mean()), 4), "std": round(float(mf1s.std(ddof=0)), 4)},
            "weighted_f1": {"mean": round(float(wf1s.mean()), 4), "std": round(float(wf1s.std(ddof=0)), 4)},
            "stats": {
                "total_params_m": runs[0]["stats"]["total_params_m"],
                "trainable_params_m": runs[0]["stats"]["trainable_params_m"],
                "latency_ms_mean": round(float(lats.mean()), 2),
            },
        }

    summary_payload = {
        "session_id": SESSION_ID,
        "dataset": "intel",
        "completed_units": completed_units,
        "aggregated": agg_summary,
    }
    with open(res_dir / "s1_train_a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Refresh zip archives for one-click download on Kaggle/Colab
    shutil.make_archive(str(root / "s1_train_a_results"), "zip", str(res_dir))


# ---------------------------------------------------------------------------
# Main Runner with Time Guard, Dry Run, and Unit Resume
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session S1-train-a: Intel Full-Dataset Training (ResNet50, EfficientNet-B0, ConvNeXt-Tiny)")
    p.add_argument("--data_dir", type=str, default=None, help="Optional path to Intel dataset root")
    p.add_argument("--out_root", type=str, default=None, help="Optional output root directory")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(TIMM_REGISTRY.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--budget_minutes", type=float, default=150.0, help="Hard time guard in minutes (default 150 = 2.5h)")
    p.add_argument("--dry_run", action="store_true", help="Run 1 tiny batch per model and print estimated total wall time")
    p.add_argument("--cache_pretrained", action="store_true", help="Pre-cache all pretrained weights into pretrained_cache/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_t0 = time.perf_counter()
    budget_sec = args.budget_minutes * 60.0

    env_name, root, res_dir, ckpt_dir, cache_dir = detect_env_dirs(args.out_root)
    sync_existing_units_from_kaggle_input(res_dir, ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_meta = get_env_metadata(env_name)

    print("=" * 76)
    print(f"SESSION {SESSION_ID} | Env: {env_name} | Device: {device} ({env_meta['gpu_name']})")
    print(f"Output Root: {root} | Time Budget: {args.budget_minutes:.1f} min | Dry Run: {args.dry_run}")
    print(f"Models: {args.models} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print("=" * 76)

    if args.cache_pretrained:
        cache_all_pretrained_weights(cache_dir)

    if args.dry_run:
        print("\n[DRY RUN] Measuring per-batch and per-epoch speed across models...")
        train_loader, val_loader, test_loader, class_names = create_intel_loaders(
            args.data_dir, img_size=224, batch_size=args.batch_size, num_workers=args.num_workers, seed=0
        )
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        n_test_batches = len(test_loader)
        print(
            f"[DRY RUN] Intel Splits Verified -> Train: {len(train_loader.dataset)} ({n_train_batches} batches) | "
            f"Val: {len(val_loader.dataset)} ({n_val_batches} batches) | Test: {len(test_loader.dataset)} ({n_test_batches} batches)"
        )

        total_est_sec = 0.0
        for m in args.models:
            t0 = time.perf_counter()
            dummy_ckpt = ckpt_dir / f"_dryrun_{m}.pth"
            _, two_batch_sec = train_and_evaluate_unit(
                model_name=m,
                seed=0,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=class_names,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                ckpt_path=dummy_ckpt,
                dry_run=True,
            )
            # Estimate full 6-epoch unit time from 2-batch dry run
            sec_per_batch = max(0.02, two_batch_sec / 6.0)
            est_ep_sec = (n_train_batches + n_val_batches) * sec_per_batch
            est_unit_sec = args.epochs * est_ep_sec + n_test_batches * sec_per_batch
            est_model_3seeds_sec = 3 * est_unit_sec
            total_est_sec += est_model_3seeds_sec
            print(
                f"  [DRY RUN | {m}] 2-batch check: {time.perf_counter()-t0:.1f}s | "
                f"Est. 1 unit ({args.epochs} ep): {est_unit_sec/60:.1f} min | "
                f"Est. 3 seeds: {est_model_3seeds_sec/60:.1f} min"
            )

        print("-" * 76)
        print(
            f"[DRY RUN COMPLETE] Estimated total time for {len(args.models)*len(args.seeds)} units: "
            f"{total_est_sec/60:.1f} min ({total_est_sec/3600:.2f} hours) vs. budget {args.budget_minutes:.1f} min."
        )
        print("Ready for full execution!")
        return

    # Build unit queue: (model_name, seed)
    units: List[Tuple[str, int]] = [(m, s) for m in args.models for s in args.seeds]
    total_units = len(units)
    unit_durations: Dict[str, List[float]] = {m: [] for m in args.models}
    all_durations: List[float] = []

    # Cache DataLoaders by seed so we only build the stratified split once per seed
    loaders_by_seed: Dict[int, Tuple[DataLoader, DataLoader, DataLoader, List[str]]] = {}

    for idx, (model_name, seed) in enumerate(units, start=1):
        unit_id = f"s1_train_intel_{model_name}_seed{seed}"
        json_path = res_dir / f"{unit_id}.json"
        ckpt_path = ckpt_dir / f"best_{model_name}_intel_seed{seed}.pth"

        # Rule b: Resume — skip units whose JSON already exists
        if json_path.is_file() and ckpt_path.is_file():
            print(f"[UNIT {idx}/{total_units}] {unit_id} -> SKIPPED (already exists at {json_path.name})")
            continue

        # Rule c: Time guard — check before starting a new unit
        elapsed_sec = time.perf_counter() - session_t0
        remaining_sec = budget_sec - elapsed_sec
        if unit_durations[model_name]:
            est_next_sec = float(np.mean(unit_durations[model_name]))
        elif all_durations:
            est_next_sec = float(np.mean(all_durations))
        else:
            est_next_sec = 360.0  # Conservative 6-min initial estimate on T4

        if idx > 1 and est_next_sec > remaining_sec:
            rem_units = [f"s1_train_intel_{m}_seed{s}" for (m, s) in units[idx - 1 :]]
            print("\n" + "!" * 76)
            print(
                f"[TIME GUARD] Stopping cleanly before {unit_id}: estimated unit duration "
                f"({est_next_sec/60:.1f}m) exceeds remaining budget ({remaining_sec/60:.1f}m)."
            )
            print(f"[TIME GUARD] Remaining units ({len(rem_units)}): {rem_units}")
            print("!" * 76)
            break

        print(f"\n>>> [UNIT {idx}/{total_units}] Starting {unit_id} (Elapsed: {elapsed_sec/60:.1f}m, Est: {est_next_sec/60:.1f}m) <<<")
        unit_t0 = time.perf_counter()

        if seed not in loaders_by_seed:
            loaders_by_seed[seed] = create_intel_loaders(
                args.data_dir,
                img_size=224,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed,
            )
        train_loader, val_loader, test_loader, class_names = loaders_by_seed[seed]

        res, _ = train_and_evaluate_unit(
            model_name=model_name,
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            ckpt_path=ckpt_path,
            dry_run=False,
        )

        unit_sec = time.perf_counter() - unit_t0
        unit_durations[model_name].append(unit_sec)
        all_durations.append(unit_sec)

        unit_payload = {
            "unit_id": unit_id,
            "session_id": SESSION_ID,
            "dataset": "intel",
            "model": model_name,
            "timm_id": TIMM_REGISTRY[model_name],
            "seed": seed,
            "environment": env_meta,
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "val_ratio": 0.15,
                "num_train": len(train_loader.dataset),
                "num_val": len(val_loader.dataset),
                "num_test": len(test_loader.dataset),
                "train_amp": device.type == "cuda",
                "eval_fp32": True,
            },
            "history": res["history"],
            "test_metrics": {
                "accuracy": res["test_metrics"]["accuracy"],
                "macro_f1": res["test_metrics"]["macro_f1"],
                "weighted_f1": res["test_metrics"]["weighted_f1"],
                "loss": res["test_metrics"]["loss"],
                "per_class_accuracy": res["test_metrics"]["per_class_accuracy"],
                "confusion_matrix": res["test_metrics"]["confusion_matrix"],
            },
            "stats": res["stats"],
            "checkpoint": {
                "path": str(ckpt_path.name),
                "fp16_state_dict": True,
                "size_mb": res["checkpoint_size_mb"],
            },
            "predictions": {
                "y_true": res["test_metrics"]["y_true"],
                "y_pred": res["test_metrics"]["y_pred"],
            },
            "elapsed_seconds": round(unit_sec, 2),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unit_payload, f, indent=2)

        update_summary_and_zips(root, res_dir, ckpt_dir, args.models, args.seeds)

        total_elapsed = time.perf_counter() - session_t0
        units_left = total_units - idx
        eta_sec = units_left * float(np.mean(all_durations))
        print(
            f"[UNIT {idx}/{total_units} DONE] {unit_id} | Test Acc: {res['test_metrics']['accuracy']*100:.2f}% | "
            f"Macro-F1: {res['test_metrics']['macro_f1']*100:.2f}% | Ckpt: {res['checkpoint_size_mb']} MB | "
            f"Unit: {unit_sec/60:.1f}m | Elapsed: {total_elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m"
        )

    update_summary_and_zips(root, res_dir, ckpt_dir, args.models, args.seeds)
    print("\n" + "=" * 76)
    print(f"SESSION {SESSION_ID} FINISHED in {(time.perf_counter()-session_t0)/60:.1f} minutes.")
    print(f"Results saved in:     {res_dir}")
    print(f"Checkpoints saved in: {ckpt_dir}")
    print(f"Downloadable zip:     {root / 's1_train_a_results.zip'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
