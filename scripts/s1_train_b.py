#!/usr/bin/env python3
"""
Session S1-train-b: Unified Full-Dataset Intel Natural Scenes Training & Clean Evaluation
========================================================================================
Models (3): swin_tiny (6 epochs), clip_linear (5 epochs), clip_zeroshot (0 epochs, deterministic baseline)
Seeds  (3): 0, 1, 2
Total Units: 7 units:
  - swin_tiny     x seeds {0, 1, 2} (3 units, 6 epochs, lr=2e-4)
  - clip_linear   x seeds {0, 1, 2} (3 units, 5 epochs, lr=1e-3, frozen ViT-B/16 + linear probe head)
  - clip_zeroshot x seed 0 (+ deterministic mirror JSONs for seeds 1, 2 so all 6 models have 3-seed records)

Seamless S1-train-a Integration:
- Automatically scans /kaggle/input/* for attached S1-train-a outputs (JSONs + FP16 .pth checkpoints)
  and copies them into /kaggle/working/results/ and /kaggle/working/checkpoints/.
- Consequently, when S1-train-b finishes, its /kaggle/working/ directory contains the COMPLETE
  set of all 6 architectures x 3 seeds (18 unit JSONs + all 16 FP16 checkpoints + unified summary
  results/s1_train_full_intel_6models_summary.json), so future sessions (S1a, S1b, S2, S4, S7)
  only need to attach S1-train-b's output!
"""

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import timm

# ---------------------------------------------------------------------------
# Constants & Prompt Templates
# ---------------------------------------------------------------------------
SESSION_ID = "S1-train-b"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PROMPT_TEMPLATES: Sequence[str] = (
    "a photo of a {}.",
    "a photo of the {} scene.",
    "an image of a {}.",
    "a natural scene showing {}.",
    "a photograph of a {} environment.",
)

MODEL_ID_MAP: Dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "clip_zeroshot": "open_clip:ViT-B-16(openai)",
    "clip_linear": "open_clip:ViT-B-16(openai)+LinearProbe",
}

DEFAULT_MODELS = ["swin_tiny", "clip_linear", "clip_zeroshot"]
DEFAULT_SEEDS = [0, 1, 2]
ALL_SIX_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "swin_tiny",
    "clip_zeroshot",
    "clip_linear",
]


# ---------------------------------------------------------------------------
# Environment & Reproducibility Helpers
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

    # Check if an attached Kaggle input has pretrained_cache
    used_input_cache = False
    k_in = Path("/kaggle/input")
    if k_in.is_dir():
        for ds_dir in sorted(k_in.iterdir()):
            cand = ds_dir / "pretrained_cache"
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


def sync_existing_units_from_kaggle_input(
    res_dir: Path, ckpt_dir: Path, copy_checkpoints: bool = True
) -> None:
    """Copy unit JSONs and FP16 checkpoints from attached S1-train-a notebook output into /kaggle/working."""
    k_in = Path("/kaggle/input")
    if not k_in.is_dir():
        return
    copied_jsons = 0
    copied_ckpts = 0
    for ds_dir in sorted(k_in.iterdir()):
        if not ds_dir.is_dir():
            continue
        for cand_res in [ds_dir / "results", ds_dir]:
            if cand_res.is_dir():
                for jf in cand_res.glob("s1_train_*.json"):
                    dst = res_dir / jf.name
                    if not dst.is_file():
                        shutil.copy2(jf, dst)
                        copied_jsons += 1
        for cand_ckpt in [ds_dir / "checkpoints", ds_dir]:
            if cand_ckpt.is_dir():
                for cf in cand_ckpt.glob("best_*_intel_seed*.pth"):
                    dst = ckpt_dir / cf.name
                    if not dst.is_file():
                        if copy_checkpoints:
                            shutil.copy2(cf, dst)
                        else:
                            try:
                                os.symlink(cf.resolve(), dst)
                            except Exception:
                                shutil.copy2(cf, dst)
                        copied_ckpts += 1
    if copied_jsons or copied_ckpts:
        print(
            f"[SYNC] Imported {copied_jsons} unit JSONs and {copied_ckpts} FP16 checkpoints "
            f"from attached Kaggle input into {res_dir.parent}!"
        )


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

    try:
        import open_clip  # type: ignore

        oc_ver = getattr(open_clip, "__version__", "installed")
    except Exception:
        oc_ver = "not_loaded"

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
        "open_clip_version": oc_ver,
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

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for p in sorted(kaggle_input.iterdir()):
            if p.is_dir():
                search_roots.append(p)

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
# CLIP ViT-B/16 Wrapper & Swin-Tiny Builder
# ---------------------------------------------------------------------------
def _clean_class_label(raw_name: str) -> str:
    return raw_name.strip("/").split("/")[-1].replace("_", " ")


class CLIPSceneClassifier(nn.Module):
    """Wraps open_clip ViT-B/16 for Zero-Shot or Linear-Probe scene classification."""

    def __init__(
        self,
        class_names: List[str],
        mode: str = "zeroshot",
        model_arch: str = "ViT-B-16",
        pretrained: str = "openai",
    ):
        super().__init__()
        import open_clip  # type: ignore

        self.mode = mode.lower()
        self.class_names = class_names
        self.num_classes = len(class_names)

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_arch, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_arch)
        self.visual = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp().detach()

        im_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        im_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        cl_mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        cl_std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("im_mean", im_mean)
        self.register_buffer("im_std", im_std)
        self.register_buffer("cl_mean", cl_mean)
        self.register_buffer("cl_std", cl_std)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.visual(dummy).shape[-1]

        for p in self.visual.parameters():
            p.requires_grad = False

        text_weights = self._build_zeroshot_weights(clip_model, tokenizer, class_names)
        if self.mode == "zeroshot":
            self.register_buffer("text_Features", text_weights)
            self.head = None
        elif self.mode == "linear":
            self.head = nn.Linear(feat_dim, self.num_classes)
            with torch.no_grad():
                self.head.weight.copy_(text_weights.T * float(self.logit_scale.item()))
                self.head.bias.zero_()
        else:
            raise ValueError(f"Unknown CLIP mode '{mode}'.")

    @torch.no_grad()
    def _build_zeroshot_weights(
        self, clip_model: nn.Module, tokenizer, class_names: List[str]
    ) -> torch.Tensor:
        clip_model.eval()
        zeroshot_weights = []
        for cls_name in class_names:
            clean_name = _clean_class_label(cls_name)
            texts = [tpl.format(clean_name) for tpl in PROMPT_TEMPLATES]
            tokens = tokenizer(texts)
            class_embeddings = clip_model.encode_text(tokens)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            mean_embedding = class_embeddings.mean(dim=0)
            mean_embedding = F.normalize(mean_embedding, dim=-1)
            zeroshot_weights.append(mean_embedding)
        return torch.stack(zeroshot_weights, dim=1)

    def _to_clip_norm(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x * self.im_std + self.im_mean
        return (rgb - self.cl_mean) / self.cl_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clip = self._to_clip_norm(x)
        img_feats = self.visual(x_clip)
        img_feats = F.normalize(img_feats, dim=-1)
        if self.mode == "zeroshot":
            return float(self.logit_scale.item()) * (img_feats @ self.text_Features)
        return self.head(img_feats)


def build_s1b_model(
    model_name: str, num_classes: int, class_names: List[str], drop_rate: float = 0.2
) -> nn.Module:
    key = model_name.lower()
    if key == "swin_tiny":
        return timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    if key == "clip_zeroshot":
        return CLIPSceneClassifier(class_names=class_names, mode="zeroshot")
    if key == "clip_linear":
        return CLIPSceneClassifier(class_names=class_names, mode="linear")
    raise ValueError(f"Unsupported model '{model_name}' for S1-train-b.")


# ---------------------------------------------------------------------------
# Strict FP32 Evaluation & AMP Training
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_fp32(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
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
    set_seed(seed)
    num_classes = len(class_names)
    model = build_s1b_model(model_name, num_classes=num_classes, class_names=class_names).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    max_batches = 2 if dry_run else None

    # Zero-shot model (0 trainable parameters)
    if not trainable_params or epochs <= 0:
        t0 = time.perf_counter()
        zs_crit = nn.CrossEntropyLoss()  # Unsmoothed CE for zero-shot reference loss
        val_res = evaluate_fp32(
            model, val_loader, device, class_names, criterion=zs_crit, max_batches=max_batches
        )
        test_res = evaluate_fp32(
            model, test_loader, device, class_names, criterion=zs_crit, max_batches=max_batches
        )
        stats = measure_fp32_stats(model, device, img_size=224)
        ep_sec = time.perf_counter() - t0

        history = {
            "train_loss": [val_res["loss"]],
            "train_acc": [val_res["accuracy"]],
            "val_loss": [val_res["loss"]],
            "val_acc": [val_res["accuracy"]],
            "best_epoch": 1,
            "best_val_acc": val_res["accuracy"],
        }
        print(
            f"  [{model_name} | seed={seed}] Zero-Shot Eval ({ep_sec:.1f}s) | "
            f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy']*100:.2f}% | "
            f"Test Acc: {test_res['accuracy']*100:.2f}%"
        )

        ckpt_size_mb = 0.0
        if not dry_run:
            fp16_state = {
                k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
                for k, v in model.state_dict().items()
            }
            ckpt_payload = {
                "session_id": SESSION_ID,
                "dataset": "intel",
                "model_name": model_name,
                "model_id": MODEL_ID_MAP[model_name],
                "seed": seed,
                "num_classes": num_classes,
                "class_names": class_names,
                "best_epoch": 1,
                "best_val_acc": val_res["accuracy"],
                "test_accuracy": test_res["accuracy"],
                "macro_f1": test_res["macro_f1"],
                "fp16_state_dict": True,
                "model_state_dict": fp16_state,
            }
            torch.save(ckpt_payload, ckpt_path)
            ckpt_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

        return {
            "history": history,
            "test_metrics": test_res,
            "stats": stats,
            "checkpoint_size_mb": ckpt_size_mb,
        }, ep_sec

    # Supervised / Linear-Probe training
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 1,
        "best_val_acc": -1.0,
    }
    best_fp16_state: Dict[str, torch.Tensor] = {}

    run_epochs = 1 if dry_run else epochs

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
            "model_id": MODEL_ID_MAP[model_name],
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
# McNemar + Holm-Bonferroni & 6-Model Unified Summary
# ---------------------------------------------------------------------------
def compute_mcnemar_pair(
    y_true: Sequence[int], y_a: Sequence[int], y_b: Sequence[int]
) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.int64)
    pa = np.asarray(y_a, dtype=np.int64) == yt
    pb = np.asarray(y_b, dtype=np.int64) == yt

    n11 = int(np.sum(pa & pb))
    n10 = int(np.sum(pa & (~pb)))
    n01 = int(np.sum((~pa) & pb))
    n00 = int(np.sum((~pa) & (~pb)))
    disc = n10 + n01

    if disc == 0:
        chi2_cc = 0.0
        p_chi2 = 1.0
        p_exact = 1.0
    else:
        diff = abs(n10 - n01)
        chi2_cc = ((max(0.0, diff - 1.0)) ** 2) / disc
        p_chi2 = float(math.erfc(math.sqrt(chi2_cc / 2.0)))
        k = min(n10, n01)
        p_sum = sum(math.comb(disc, i) * (0.5 ** disc) for i in range(k + 1))
        p_exact = min(1.0, 2.0 * p_sum)

    return {
        "n11_both_correct": n11,
        "n10_a_only_correct": n10,
        "n01_b_only_correct": n01,
        "n00_both_wrong": n00,
        "discordant_pairs": disc,
        "chi2_cc": round(float(chi2_cc), 4),
        "p_value_chi2": round(float(p_chi2), 6),
        "p_value_exact": round(float(p_exact), 6),
        "significant_unadjusted_005": bool(p_exact < 0.05),
    }


def apply_holm_bonferroni(pair_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Add Holm-Bonferroni adjusted p-values across all M pairwise comparisons."""
    items = sorted(pair_dict.items(), key=lambda kv: kv[1]["p_value_exact"])
    m_total = len(items)
    running_max = 0.0
    for rank_idx, (key, val) in enumerate(items):
        mult = m_total - rank_idx
        adj_p = min(1.0, val["p_value_exact"] * mult)
        running_max = max(running_max, adj_p)
        val["p_value_holm"] = round(float(running_max), 6)
        val["significant_holm_005"] = bool(running_max < 0.05)
    return pair_dict


def update_summaries_and_zips(
    root: Path, res_dir: Path, seeds: List[int]
) -> None:
    """Update s1_train_b_summary.json AND unified s1_train_full_intel_6models_summary.json."""
    by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ALL_SIX_MODELS}
    s1b_completed: List[str] = []
    all_completed: List[str] = []

    for m in ALL_SIX_MODELS:
        for s in seeds:
            u_path = res_dir / f"s1_train_intel_{m}_seed{s}.json"
            if u_path.is_file():
                with open(u_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                by_model[m].append(data)
                all_completed.append(data["unit_id"])
                if m in DEFAULT_MODELS:
                    s1b_completed.append(data["unit_id"])

    def build_agg(model_keys: Sequence[str]) -> Dict[str, Any]:
        agg: Dict[str, Any] = {}
        for m in model_keys:
            runs = by_model.get(m, [])
            if not runs:
                continue
            accs = np.array([r["test_metrics"]["accuracy"] for r in runs], dtype=np.float64)
            mf1s = np.array([r["test_metrics"]["macro_f1"] for r in runs], dtype=np.float64)
            wf1s = np.array([r["test_metrics"]["weighted_f1"] for r in runs], dtype=np.float64)
            lats = np.array([r["stats"]["latency_ms"] for r in runs], dtype=np.float64)
            agg[m] = {
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
        return agg

    # Save s1_train_b_summary.json
    s1b_summary = {
        "session_id": SESSION_ID,
        "dataset": "intel",
        "completed_units": s1b_completed,
        "aggregated": build_agg(DEFAULT_MODELS),
    }
    with open(res_dir / "s1_train_b_summary.json", "w", encoding="utf-8") as f:
        json.dump(s1b_summary, f, indent=2)

    # Compute pairwise McNemar per seed (and pooled) across all models present
    mcnemar_by_seed: Dict[str, Any] = {}
    for s in seeds:
        seed_models = [
            m for m in ALL_SIX_MODELS if any(r["seed"] == s for r in by_model.get(m, []))
        ]
        if len(seed_models) >= 2:
            y_true = None
            preds_map = {}
            for m in seed_models:
                rec = next(r for r in by_model[m] if r["seed"] == s)
                y_true = rec["predictions"]["y_true"]
                preds_map[m] = rec["predictions"]["y_pred"]
            pairs_s: Dict[str, Dict[str, Any]] = {}
            for i in range(len(seed_models)):
                for j in range(i + 1, len(seed_models)):
                    ma, mb = seed_models[i], seed_models[j]
                    pairs_s[f"{ma}_vs_{mb}"] = compute_mcnemar_pair(
                        y_true, preds_map[ma], preds_map[mb]
                    )
            mcnemar_by_seed[f"seed_{s}"] = apply_holm_bonferroni(pairs_s)

    full_summary = {
        "session_id": "S1-train-unified-6models",
        "dataset": "intel",
        "completed_units": all_completed,
        "aggregated": build_agg(ALL_SIX_MODELS),
        "mcnemar_by_seed": mcnemar_by_seed,
    }
    with open(res_dir / "s1_train_full_intel_6models_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    shutil.make_archive(str(root / "s1_train_b_results"), "zip", str(res_dir))


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session S1-train-b: Intel Full-Dataset Training (Swin-Tiny, CLIP-Linear, CLIP-ZeroShot)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--swin_epochs", type=int, default=6)
    p.add_argument("--clip_linear_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--swin_lr", type=float, default=2e-4)
    p.add_argument("--clip_linear_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--budget_minutes", type=float, default=150.0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_t0 = time.perf_counter()
    budget_sec = args.budget_minutes * 60.0

    env_name, root, res_dir, ckpt_dir, _ = detect_env_dirs(args.out_root)
    sync_existing_units_from_kaggle_input(res_dir, ckpt_dir, copy_checkpoints=not args.dry_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_meta = get_env_metadata(env_name)

    print("=" * 76)
    print(f"SESSION {SESSION_ID} | Env: {env_name} | Device: {device} ({env_meta['gpu_name']})")
    print(f"Output Root: {root} | Time Budget: {args.budget_minutes:.1f} min | Dry Run: {args.dry_run}")
    print(f"Models: {args.models} | Seeds: {args.seeds}")
    print("=" * 76)

    if args.dry_run:
        print("\n[DRY RUN] Measuring per-batch speed across Swin-Tiny, CLIP-Linear, and CLIP-ZeroShot...")
        train_loader, val_loader, test_loader, class_names = create_intel_loaders(
            args.data_dir, img_size=224, batch_size=args.batch_size, num_workers=args.num_workers, seed=0
        )
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        n_test_batches = len(test_loader)

        total_est_sec = 0.0
        for m in args.models:
            t0 = time.perf_counter()
            ep = 0 if m == "clip_zeroshot" else (args.clip_linear_epochs if m == "clip_linear" else args.swin_epochs)
            lr = args.clip_linear_lr if m == "clip_linear" else args.swin_lr
            dummy_ckpt = ckpt_dir / f"_dryrun_{m}.pth"
            _, two_batch_sec = train_and_evaluate_unit(
                model_name=m,
                seed=0,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=class_names,
                device=device,
                epochs=ep,
                lr=lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                ckpt_path=dummy_ckpt,
                dry_run=True,
            )
            if m == "clip_zeroshot":
                sec_per_batch = max(0.02, two_batch_sec / 4.0)
                est_unit_sec = (n_val_batches + n_test_batches) * sec_per_batch
                est_model_sec = est_unit_sec
                n_u = 1
            else:
                sec_per_batch = max(0.02, two_batch_sec / 6.0)
                est_ep_sec = (n_train_batches + n_val_batches) * sec_per_batch
                est_unit_sec = ep * est_ep_sec + n_test_batches * sec_per_batch
                est_model_sec = len(args.seeds) * est_unit_sec
                n_u = len(args.seeds)
            total_est_sec += est_model_sec
            print(
                f"  [DRY RUN | {m}] 2-batch check: {time.perf_counter()-t0:.1f}s | "
                f"Est. 1 unit ({ep} ep): {est_unit_sec/60:.1f} min | "
                f"Est. {n_u} unit(s): {est_model_sec/60:.1f} min"
            )

        print("-" * 76)
        print(
            f"[DRY RUN COMPLETE] Estimated total time for S1-train-b: "
            f"{total_est_sec/60:.1f} min ({total_est_sec/3600:.2f} hours) vs. budget {args.budget_minutes:.1f} min."
        )
        print("Ready for full execution!")
        return

    # Build unit queue: swin_tiny (seeds 0,1,2), clip_linear (seeds 0,1,2), clip_zeroshot (seed 0)
    units: List[Tuple[str, int]] = []
    for m in args.models:
        if m == "clip_zeroshot":
            units.append((m, 0))
        else:
            for s in args.seeds:
                units.append((m, s))

    total_units = len(units)
    unit_durations: Dict[str, List[float]] = {m: [] for m in args.models}
    all_durations: List[float] = []
    loaders_by_seed: Dict[int, Tuple[DataLoader, DataLoader, DataLoader, List[str]]] = {}

    for idx, (model_name, seed) in enumerate(units, start=1):
        unit_id = f"s1_train_intel_{model_name}_seed{seed}"
        json_path = res_dir / f"{unit_id}.json"
        ckpt_path = ckpt_dir / f"best_{model_name}_intel_seed{seed}.pth"

        if json_path.is_file() and ckpt_path.is_file():
            print(f"[UNIT {idx}/{total_units}] {unit_id} -> SKIPPED (already exists at {json_path.name})")
            continue

        elapsed_sec = time.perf_counter() - session_t0
        remaining_sec = budget_sec - elapsed_sec
        if unit_durations[model_name]:
            est_next_sec = float(np.mean(unit_durations[model_name]))
        elif model_name == "clip_zeroshot":
            est_next_sec = 90.0
        elif all_durations:
            est_next_sec = float(np.mean(all_durations))
        else:
            est_next_sec = 420.0

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

        ep = 0 if model_name == "clip_zeroshot" else (
            args.clip_linear_epochs if model_name == "clip_linear" else args.swin_epochs
        )
        lr = 0.0 if model_name == "clip_zeroshot" else (
            args.clip_linear_lr if model_name == "clip_linear" else args.swin_lr
        )

        res, _ = train_and_evaluate_unit(
            model_name=model_name,
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            epochs=ep,
            lr=lr,
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
            "model_id": MODEL_ID_MAP[model_name],
            "seed": seed,
            "environment": env_meta,
            "hyperparameters": {
                "epochs": ep,
                "batch_size": args.batch_size,
                "lr": lr,
                "weight_decay": args.weight_decay if ep > 0 else 0.0,
                "label_smoothing": args.label_smoothing if ep > 0 else 0.0,
                "val_ratio": 0.15,
                "num_train": len(train_loader.dataset),
                "num_val": len(val_loader.dataset),
                "num_test": len(test_loader.dataset),
                "train_amp": (device.type == "cuda") and (ep > 0),
                "eval_fp32": True,
                "prompt_templates": list(PROMPT_TEMPLATES) if "clip" in model_name else None,
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

        # If clip_zeroshot seed 0 just finished, also write deterministic mirror JSONs for seeds 1 and 2
        if model_name == "clip_zeroshot" and seed == 0:
            for extra_s in args.seeds:
                if extra_s == 0:
                    continue
                extra_id = f"s1_train_intel_clip_zeroshot_seed{extra_s}"
                extra_json = res_dir / f"{extra_id}.json"
                mirror_payload = dict(unit_payload)
                mirror_payload["unit_id"] = extra_id
                mirror_payload["seed"] = extra_s
                mirror_payload["deterministic_mirror_of_seed0"] = True
                with open(extra_json, "w", encoding="utf-8") as f:
                    json.dump(mirror_payload, f, indent=2)

        update_summaries_and_zips(root, res_dir, args.seeds)

        total_elapsed = time.perf_counter() - session_t0
        units_left = total_units - idx
        eta_sec = units_left * float(np.mean(all_durations))
        print(
            f"[UNIT {idx}/{total_units} DONE] {unit_id} | Test Acc: {res['test_metrics']['accuracy']*100:.2f}% | "
            f"Macro-F1: {res['test_metrics']['macro_f1']*100:.2f}% | Ckpt: {res['checkpoint_size_mb']} MB | "
            f"Unit: {unit_sec/60:.1f}m | Elapsed: {total_elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m"
        )

    update_summaries_and_zips(root, res_dir, args.seeds)
    print("\n" + "=" * 76)
    print(f"SESSION {SESSION_ID} FINISHED in {(time.perf_counter()-session_t0)/60:.1f} minutes.")
    print(f"Results saved in:     {res_dir}")
    print(f"Checkpoints saved in: {ckpt_dir}")
    print(f"Downloadable zip:     {root / 's1_train_b_results.zip'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
