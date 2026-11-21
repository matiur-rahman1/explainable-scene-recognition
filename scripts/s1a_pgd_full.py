#!/usr/bin/env python3
"""
Session S1a: Full-Test (N=3,000) Strict FP32 Adversarial Robustness on Intel Natural Scenes
===========================================================================================
Models (3): resnet50, efficientnet_b0, convnext_tiny
Seeds  (3): 0, 1, 2
Total Units: 9 units (1 unit = 1 (dataset='intel', model, seed))

Protocol (Hard Rules 3 & 4):
- Evaluates on ALL 3,000 Intel test images (zero 600-image subset approximation!).
- Runs in strict FP32 (`torch.amp.autocast(..., enabled=False)`) — NO AMP during attacks or eval!
- Attacks evaluated per unit at eps in {0.0, 0.01, 0.03, 0.05}:
  1. PGD-10 (10-step L_inf Projected Gradient Descent, random uniform start, step_size = 2.5 * eps / 10)
  2. FGSM (1-step true-label Fast Gradient Sign Method)
  3. FGSM_pred (1-step predicted-label FGSM to diagnose label leaking vs. loss-surface overshooting)
- Checkpoint loading:
  - Recursively searches `/kaggle/working/checkpoints` and `/kaggle/input/**` (`rglob`) for
    `best_<model>_intel_seed<seed>.pth` (from S1-train-a).
  - If a checkpoint is not found in `/kaggle/input/**`, automatically trains the 6-epoch S1-train-a
    checkpoint first so the session never fails even if an input notebook was not attached!
"""

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import timm

# ---------------------------------------------------------------------------
# Constants & Registry
# ---------------------------------------------------------------------------
SESSION_ID = "S1a"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TIMM_REGISTRY: Dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
}

DEFAULT_MODELS = ["resnet50", "efficientnet_b0", "convnext_tiny"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_EPSILONS = (0.0, 0.01, 0.03, 0.05)


# ---------------------------------------------------------------------------
# Environment & Recursive Kaggle Input Discovery
# ---------------------------------------------------------------------------
def detect_env_dirs(
    custom_out_root: Optional[str] = None,
) -> Tuple[str, Path, Path, Path, Path]:
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
                print(f"[WARN] Could not mount Colab Drive ({e}); using /content/scene_results")
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

    # Recursively check if an attached Kaggle input has pretrained_cache
    used_input_cache = False
    k_in = Path("/kaggle/input")
    if k_in.is_dir():
        for cand in k_in.rglob("pretrained_cache"):
            if cand.is_dir():
                os.environ.setdefault("HF_HOME", str(cand / "hf"))
                os.environ.setdefault("TORCH_HOME", str(cand / "torch"))
                os.environ.setdefault("TIMM_CACHE_DIR", str(cand / "timm"))
                used_input_cache = True
                print(f"[CACHE] Using attached pretrained_cache from {cand}")
                break

    if not used_input_cache:
        os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
        os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
        os.environ.setdefault("TIMM_CACHE_DIR", str(cache_dir / "timm"))

    return env_name, root, res_dir, ckpt_dir, cache_dir


def discover_checkpoints_and_results(res_dir: Path, ckpt_dir: Path) -> Dict[str, Path]:
    """Recursively search /kaggle/working and /kaggle/input/** for S1-train checkpoints and S1a JSONs."""
    found_ckpts: Dict[str, Path] = {}

    # 1. Check ckpt_dir first
    for cf in ckpt_dir.glob("best_*_intel_seed*.pth"):
        found_ckpts[cf.name] = cf

    # 2. Recursively search /kaggle/input/**
    k_in = Path("/kaggle/input")
    if k_in.is_dir():
        for cf in k_in.rglob("best_*_intel_seed*.pth"):
            if cf.name not in found_ckpts:
                found_ckpts[cf.name] = cf
        for jf in k_in.rglob("s1a_pgd_full_intel_*.json"):
            dst = res_dir / jf.name
            if not dst.is_file():
                shutil.copy2(jf, dst)
        for jf in k_in.rglob("s1_train_intel_*.json"):
            dst = res_dir / jf.name
            if not dst.is_file():
                shutil.copy2(jf, dst)

    print(f"[DISCOVERY] Found {len(found_ckpts)} saved checkpoints across working & input directories:")
    for name, path in sorted(found_ckpts.items()):
        print(f"  - {name} -> {path}")
    return found_ckpts


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
    for cwd in [Path.cwd(), Path("/kaggle/working/explainable-scene-recognition")]:
        if (cwd / ".git").is_dir():
            try:
                out = subprocess.check_output(
                    ["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                return out.decode("utf-8").strip()
            except Exception:
                pass
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
    search_roots: List[Path] = []
    if data_dir:
        search_roots.append(Path(data_dir))

    k_in = Path("/kaggle/input")
    if k_in.is_dir():
        for p in sorted(k_in.iterdir()):
            if p.is_dir():
                search_roots.append(p)
        for seg_tr in k_in.rglob("seg_train"):
            if seg_tr.is_dir():
                search_roots.append(seg_tr.parent)

    search_roots.extend([Path("./data/intel"), Path("./data")])

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
# Model Loading / Fallback Training
# ---------------------------------------------------------------------------
def build_timm_backbone(
    model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2
) -> nn.Module:
    key = model_name.lower()
    if key not in TIMM_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}' for S1a.")
    return timm.create_model(
        TIMM_REGISTRY[key],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def load_or_train_checkpoint(
    model_name: str,
    seed: int,
    found_ckpts: Dict[str, Path],
    ckpt_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    epochs: int = 6,
    lr: float = 2e-4,
) -> Tuple[nn.Module, str]:
    """Load saved FP16 checkpoint into FP32 model, or train if not found in /kaggle/input."""
    ckpt_name = f"best_{model_name}_intel_seed{seed}.pth"
    num_classes = len(class_names)

    if ckpt_name in found_ckpts and found_ckpts[ckpt_name].is_file():
        src_path = found_ckpts[ckpt_name]
        print(f"  [CKPT] Loading saved checkpoint: {src_path}")
        model = build_timm_backbone(model_name, num_classes=num_classes, pretrained=False)
        ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
        raw_sd = ckpt.get("model_state_dict", ckpt)
        fp32_sd = {
            k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in raw_sd.items()
        }
        model.load_state_dict(fp32_sd)
        model = model.to(device).float()
        model.eval()
        # Copy into /kaggle/working/checkpoints if it came from /kaggle/input
        dst_path = ckpt_dir / ckpt_name
        if not dst_path.is_file():
            shutil.copy2(src_path, dst_path)
        return model, str(src_path)

    print(
        f"  [WARN] {ckpt_name} not found in attached inputs! Training {model_name} (seed={seed}, {epochs} epochs) now..."
    )
    set_seed(seed)
    model = build_timm_backbone(model_name, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = -1.0
    best_fp16_state: Dict[str, torch.Tensor] = {}

    for ep in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"Train {model_name} Ep {ep}"):
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
        scheduler.step()

        model.eval()
        corr = 0
        tot = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    preds = model(images).argmax(dim=1)
                corr += int((preds == labels).sum().item())
                tot += images.size(0)
        val_acc = corr / max(1, tot)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_fp16_state = {
                k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
                for k, v in model.state_dict().items()
            }

    restored = {
        k: (v.float() if v.is_floating_point() else v) for k, v in best_fp16_state.items()
    }
    model.load_state_dict(restored)
    model = model.to(device).float()
    model.eval()

    dst_path = ckpt_dir / ckpt_name
    torch.save(
        {
            "session_id": SESSION_ID,
            "model_name": model_name,
            "seed": seed,
            "best_val_acc": round(best_val_acc, 4),
            "fp16_state_dict": True,
            "model_state_dict": best_fp16_state,
        },
        dst_path,
    )
    found_ckpts[ckpt_name] = dst_path
    return model, str(dst_path)


# ---------------------------------------------------------------------------
# Strict FP32 White-Box Attacks (FGSM, FGSM_pred, PGD-10) on Full Test Set
# ---------------------------------------------------------------------------
def _norm_bounds(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper, std


def fgsm_attack_fp32(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    use_pred_label: bool = False,
) -> torch.Tensor:
    """Strict FP32 single-step FGSM (true-label or predicted-label)."""
    if epsilon <= 0.0:
        return images.detach().clone()

    model.eval()
    device = images.device
    lower, upper, std = _norm_bounds(device)
    eps_scaled = float(epsilon) / std

    with torch.amp.autocast(device_type=device.type, enabled=False):
        adv = images.detach().float().clone().requires_grad_(True)
        logits = model(adv)
        target = logits.argmax(dim=1).detach() if use_pred_label else labels
        loss = F.cross_entropy(logits, target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_sign = adv.grad.detach().sign()

        with torch.no_grad():
            adv_out = adv + eps_scaled * grad_sign
            adv_out = torch.max(torch.min(adv_out, upper), lower)
    return adv_out.detach()


def pgd_attack_fp32(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    num_steps: int = 10,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Strict FP32 iterative L_inf PGD-10 with random uniform start and step_size = 2.5 * eps / 10."""
    if epsilon <= 0.0:
        return images.detach().clone()

    model.eval()
    device = images.device
    lower, upper, std = _norm_bounds(device)
    step_val = alpha if alpha is not None else (2.5 * float(epsilon) / max(1, num_steps))
    eps_scaled = float(epsilon) / std
    alpha_scaled = float(step_val) / std

    with torch.amp.autocast(device_type=device.type, enabled=False):
        orig = images.detach().float()
        delta = torch.empty_like(orig).uniform_(-1.0, 1.0) * eps_scaled
        adv = torch.max(torch.min(orig + delta, upper), lower).detach()

        for _ in range(num_steps):
            adv.requires_grad_(True)
            logits = model(adv)
            loss = F.cross_entropy(logits, labels)
            model.zero_grad(set_to_none=True)
            loss.backward()
            grad_sign = adv.grad.detach().sign()

            with torch.no_grad():
                adv = adv + alpha_scaled * grad_sign
                eta = torch.clamp(adv - orig, min=-eps_scaled, max=eps_scaled)
                adv = torch.max(torch.min(orig + eta, upper), lower).detach()

    return adv.detach()


def evaluate_full_test_attacks_fp32(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    seed: int,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    pgd_steps: int = 10,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate Clean, FGSM, FGSM_pred, and PGD-10 on all 3,000 test images in strict FP32."""
    model.eval()
    model.to(device).float()

    # 1. Clean evaluation (eps = 0.0) once
    clean_corr = 0
    clean_tot = 0
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=False):
            for b_idx, (images, labels) in enumerate(test_loader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)
                preds = model(images).argmax(dim=1)
                clean_corr += int((preds == labels).sum().item())
                clean_tot += images.size(0)
    clean_acc = round(clean_corr / max(1, clean_tot), 4)

    fgsm_res: Dict[str, float] = {"0.0": clean_acc}
    fgsm_pred_res: Dict[str, float] = {"0.0": clean_acc}
    pgd_res: Dict[str, float] = {"0.0": clean_acc}
    pgd_correct_counts: Dict[str, int] = {"0.0": clean_corr}

    nonzero_eps = [float(e) for e in epsilons if float(e) > 0.0]

    for eps in nonzero_eps:
        eps_key = str(eps)
        f_corr = 0
        fp_corr = 0
        p_corr = 0
        tot = 0

        # Lock seed before each epsilon so random starts in PGD are 100% deterministic
        set_seed(seed * 1000 + int(round(eps * 10000)))

        pbar = tqdm(
            test_loader,
            leave=False,
            desc=f"FP32 Attacks (eps={eps})",
            disable=(max_batches is not None),
        )
        for b_idx, (images, labels) in enumerate(pbar):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            # A. True-label FGSM (FP32)
            adv_fgsm = fgsm_attack_fp32(model, images, labels, epsilon=eps, use_pred_label=False)
            # B. Predicted-label FGSM_pred (FP32)
            adv_fgsm_pred = fgsm_attack_fp32(model, images, labels, epsilon=eps, use_pred_label=True)
            # C. 10-step PGD-Linf (FP32)
            adv_pgd = pgd_attack_fp32(model, images, labels, epsilon=eps, num_steps=pgd_steps)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    f_preds = model(adv_fgsm).argmax(dim=1)
                    fp_preds = model(adv_fgsm_pred).argmax(dim=1)
                    p_preds = model(adv_pgd).argmax(dim=1)

            f_corr += int((f_preds == labels).sum().item())
            fp_corr += int((fp_preds == labels).sum().item())
            p_corr += int((p_preds == labels).sum().item())
            tot += images.size(0)

        fgsm_res[eps_key] = round(f_corr / max(1, tot), 4)
        fgsm_pred_res[eps_key] = round(fp_corr / max(1, tot), 4)
        pgd_res[eps_key] = round(p_corr / max(1, tot), 4)
        pgd_correct_counts[eps_key] = p_corr

    # Diagnose monotonicity across ordered epsilons [0.0, 0.01, 0.03, 0.05]
    eps_keys = [str(float(e)) for e in sorted(epsilons)]
    pgd_seq = [pgd_res[k] for k in eps_keys]
    fgsm_seq = [fgsm_res[k] for k in eps_keys]
    fgsm_pred_seq = [fgsm_pred_res[k] for k in eps_keys]

    pgd_monotonic = all(pgd_seq[i] >= pgd_seq[i + 1] for i in range(len(pgd_seq) - 1))
    fgsm_overshoot = any(fgsm_seq[i] < fgsm_seq[i + 1] for i in range(len(fgsm_seq) - 1))
    fgsm_pred_overshoot = any(fgsm_pred_seq[i] < fgsm_pred_seq[i + 1] for i in range(len(fgsm_pred_seq) - 1))

    return {
        "num_test_images": clean_tot,
        "clean_accuracy": clean_acc,
        "fgsm_accuracy": fgsm_res,
        "fgsm_pred_accuracy": fgsm_pred_res,
        "pgd_accuracy": pgd_res,
        "pgd_correct_counts": pgd_correct_counts,
        "diagnostics": {
            "pgd_strictly_monotonic_non_increasing": pgd_monotonic,
            "fgsm_non_monotonic_overshoot": fgsm_overshoot,
            "fgsm_pred_non_monotonic_overshoot": fgsm_pred_overshoot,
        },
    }


# ---------------------------------------------------------------------------
# Summary Aggregation & Zip Sync
# ---------------------------------------------------------------------------
def update_summary_and_zips(
    root: Path, res_dir: Path, models: List[str], seeds: List[int]
) -> None:
    by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    completed_units: List[str] = []

    for m in models:
        for s in seeds:
            u_path = res_dir / f"s1a_pgd_full_intel_{m}_seed{s}.json"
            if u_path.is_file():
                with open(u_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                by_model[m].append(data)
                completed_units.append(data["unit_id"])

    agg_summary: Dict[str, Any] = {}
    for m, runs in by_model.items():
        if not runs:
            continue

        def _agg_curve(curve_key: str) -> Dict[str, Dict[str, float]]:
            out = {}
            for eps_k in runs[0][curve_key].keys():
                vals = np.array([r[curve_key][eps_k] for r in runs], dtype=np.float64)
                out[eps_k] = {
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std(ddof=0)), 4),
                }
            return out

        agg_summary[m] = {
            "num_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "num_test_images": runs[0]["num_test_images"],
            "fgsm_accuracy": _agg_curve("fgsm_accuracy"),
            "fgsm_pred_accuracy": _agg_curve("fgsm_pred_accuracy"),
            "pgd_accuracy": _agg_curve("pgd_accuracy"),
            "pgd_monotonic_all_seeds": all(
                r["diagnostics"]["pgd_strictly_monotonic_non_increasing"] for r in runs
            ),
            "fgsm_overshoot_any_seed": any(
                r["diagnostics"]["fgsm_non_monotonic_overshoot"] for r in runs
            ),
            "fgsm_pred_overshoot_any_seed": any(
                r["diagnostics"]["fgsm_pred_non_monotonic_overshoot"] for r in runs
            ),
        }

    summary_payload = {
        "session_id": SESSION_ID,
        "dataset": "intel",
        "num_test_images": 3000,
        "attack_precision": "FP32 (no AMP)",
        "pgd_steps": 10,
        "pgd_step_size": "2.5 * eps / 10",
        "completed_units": completed_units,
        "aggregated": agg_summary,
    }
    with open(res_dir / "s1a_pgd_full_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    shutil.make_archive(str(root / "s1a_results"), "zip", str(res_dir))


# ---------------------------------------------------------------------------
# CLI & Main Loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Session S1a: Full-Test (N=3000) Strict FP32 FGSM/FGSM_pred/PGD-10 on Intel"
    )
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--budget_minutes", type=float, default=150.0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_t0 = time.perf_counter()
    budget_sec = args.budget_minutes * 60.0

    env_name, root, res_dir, ckpt_dir, _ = detect_env_dirs(args.out_root)
    found_ckpts = discover_checkpoints_and_results(res_dir, ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_meta = get_env_metadata(env_name)

    print("=" * 76)
    print(f"SESSION {SESSION_ID} | Env: {env_name} | Device: {device} ({env_meta['gpu_name']})")
    print(f"Output Root: {root} | Time Budget: {args.budget_minutes:.1f} min | Dry Run: {args.dry_run}")
    print(f"Models: {args.models} | Seeds: {args.seeds} | Full Test N=3000 | Precision: Strict FP32")
    print("=" * 76)

    if args.dry_run:
        print("\n[DRY RUN] Measuring strict FP32 attack speed (FGSM + FGSM_pred + PGD-10) on 2 batches...")
        _, _, test_loader, class_names = create_intel_loaders(
            args.data_dir, img_size=224, batch_size=args.batch_size, num_workers=args.num_workers, seed=0
        )
        n_test_batches = len(test_loader)
        print(f"[DRY RUN] Intel Test Set: {len(test_loader.dataset)} images ({n_test_batches} batches of {args.batch_size})")

        total_est_sec = 0.0
        for m in args.models:
            t0 = time.perf_counter()
            model = build_timm_backbone(m, num_classes=len(class_names), pretrained=False).to(device).float()
            _ = evaluate_full_test_attacks_fp32(
                model=model,
                test_loader=test_loader,
                device=device,
                seed=0,
                epsilons=DEFAULT_EPSILONS,
                pgd_steps=args.pgd_steps,
                max_batches=2,
            )
            two_b_sec = time.perf_counter() - t0
            est_unit_sec = (n_test_batches / 2.0) * two_b_sec
            # Add fallback training time only if checkpoint is missing
            missing_cnt = sum(1 for s in args.seeds if f"best_{m}_intel_seed{s}.pth" not in found_ckpts)
            est_model_sec = len(args.seeds) * est_unit_sec + missing_cnt * 360.0
            total_est_sec += est_model_sec
            ckpt_status = f"{len(args.seeds)-missing_cnt}/{len(args.seeds)} ckpts found"
            print(
                f"  [DRY RUN | {m}] 2-batch FP32 attacks: {two_b_sec:.2f}s | "
                f"Est. 1 unit (3000 imgs): {est_unit_sec/60:.1f} min | "
                f"Est. 3 seeds ({ckpt_status}): {est_model_sec/60:.1f} min"
            )

        print("-" * 76)
        print(
            f"[DRY RUN COMPLETE] Estimated total S1a wall time for {len(args.models)*len(args.seeds)} units: "
            f"{total_est_sec/60:.1f} min ({total_est_sec/3600:.2f} hours) vs. budget {args.budget_minutes:.1f} min."
        )
        return

    units: List[Tuple[str, int]] = [(m, s) for m in args.models for s in args.seeds]
    total_units = len(units)
    unit_durations: Dict[str, List[float]] = {m: [] for m in args.models}
    all_durations: List[float] = []
    loaders_by_seed: Dict[int, Tuple[DataLoader, DataLoader, DataLoader, List[str]]] = {}

    for idx, (model_name, seed) in enumerate(units, start=1):
        unit_id = f"s1a_pgd_full_intel_{model_name}_seed{seed}"
        json_path = res_dir / f"{unit_id}.json"

        if json_path.is_file():
            print(f"[UNIT {idx}/{total_units}] {unit_id} -> SKIPPED (already exists at {json_path.name})")
            continue

        elapsed_sec = time.perf_counter() - session_t0
        remaining_sec = budget_sec - elapsed_sec
        if unit_durations[model_name]:
            est_next_sec = float(np.mean(unit_durations[model_name]))
        elif all_durations:
            est_next_sec = float(np.mean(all_durations))
        else:
            est_next_sec = 360.0

        if idx > 1 and est_next_sec > remaining_sec:
            rem_units = [f"s1a_pgd_full_intel_{m}_seed{s}" for (m, s) in units[idx - 1 :]]
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

        model, ckpt_source = load_or_train_checkpoint(
            model_name=model_name,
            seed=seed,
            found_ckpts=found_ckpts,
            ckpt_dir=ckpt_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
        )

        atk_res = evaluate_full_test_attacks_fp32(
            model=model,
            test_loader=test_loader,
            device=device,
            seed=seed,
            epsilons=DEFAULT_EPSILONS,
            pgd_steps=args.pgd_steps,
            max_batches=None,
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
            "checkpoint_source": ckpt_source,
            "environment": env_meta,
            "attack_protocol": {
                "precision": "FP32 (autocast disabled)",
                "num_test_images": atk_res["num_test_images"],
                "epsilons": list(DEFAULT_EPSILONS),
                "pgd_steps": args.pgd_steps,
                "pgd_step_size_formula": "2.5 * eps / 10",
                "pgd_random_start": True,
                "normalization_scaled_bounds": True,
            },
            "num_test_images": atk_res["num_test_images"],
            "clean_accuracy": atk_res["clean_accuracy"],
            "fgsm_accuracy": atk_res["fgsm_accuracy"],
            "fgsm_pred_accuracy": atk_res["fgsm_pred_accuracy"],
            "pgd_accuracy": atk_res["pgd_accuracy"],
            "pgd_correct_counts": atk_res["pgd_correct_counts"],
            "diagnostics": atk_res["diagnostics"],
            "elapsed_seconds": round(unit_sec, 2),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unit_payload, f, indent=2)

        update_summary_and_zips(root, res_dir, args.models, args.seeds)

        total_elapsed = time.perf_counter() - session_t0
        units_left = total_units - idx
        eta_sec = units_left * float(np.mean(all_durations))
        p_acc = atk_res["pgd_accuracy"]
        f_acc = atk_res["fgsm_accuracy"]
        print(
            f"[UNIT {idx}/{total_units} DONE] {unit_id} | Clean: {atk_res['clean_accuracy']*100:.2f}% | "
            f"FGSM(.01/.03/.05): {f_acc['0.01']*100:.2f}%/{f_acc['0.03']*100:.2f}%/{f_acc['0.05']*100:.2f}% | "
            f"PGD-10(.01/.03/.05): {p_acc['0.01']*100:.2f}%/{p_acc['0.03']*100:.2f}%/{p_acc['0.05']*100:.2f}% | "
            f"Unit: {unit_sec/60:.1f}m | Elapsed: {total_elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m"
        )

    update_summary_and_zips(root, res_dir, args.models, args.seeds)
    print("\n" + "=" * 76)
    print(f"SESSION {SESSION_ID} FINISHED in {(time.perf_counter()-session_t0)/60:.1f} minutes.")
    print(f"Results saved in: {res_dir}")
    print(f"Downloadable zip: {root / 's1a_results.zip'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
