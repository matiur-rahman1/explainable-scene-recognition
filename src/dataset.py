"""Dataset loading, preprocessing, and stratified subset utilities for scene recognition."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.utils import get_stratified_indices

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformSubset(Dataset):
    """Wraps a dataset subset with an independent transform pipeline and optional label remap."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform=None,
        label_map: Optional[Dict[int, int]] = None,
        raw_targets: Optional[Sequence[int]] = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.label_map = label_map

        base_targets = raw_targets
        if base_targets is None:
            if hasattr(dataset, "targets"):
                base_targets = getattr(dataset, "targets")
            elif hasattr(dataset, "_labels"):
                base_targets = getattr(dataset, "_labels")

        if base_targets is not None:
            if label_map is not None:
                self.targets = [label_map[int(base_targets[i])] for i in self.indices]
            else:
                self.targets = [int(base_targets[i]) for i in self.indices]
        else:
            self.targets = []

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.label_map is not None:
            label = self.label_map[int(label)]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return standard training and evaluation image transforms."""
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


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized (C, H, W) tensor to a float32 RGB image in [0, 1]."""
    img = tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy().astype(np.float32)


def _find_intel_split_dirs(root: Path) -> Tuple[Path, Path]:
    """Locate seg_train and seg_test directories inside a downloaded Kaggle directory."""
    candidates = [
        (root / "seg_train" / "seg_train", root / "seg_test" / "seg_test"),
        (root / "seg_train", root / "seg_test"),
    ]
    for train_dir, test_dir in candidates:
        if train_dir.is_dir() and test_dir.is_dir():
            subdirs = [p for p in train_dir.iterdir() if p.is_dir()]
            if subdirs and subdirs[0].name != "seg_train":
                return train_dir, test_dir
    raise FileNotFoundError(
        f"Could not locate seg_train / seg_test class folders under {root}"
    )


def _stratified_split_indices(
    targets: Sequence[int], val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split sample indices into stratified train and validation sets."""
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


def build_stratified_eval_loader(
    test_loader: DataLoader,
    max_samples: int,
    batch_size: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """Create a deterministic, class-balanced evaluation DataLoader from test_loader."""
    ds = test_loader.dataset
    targets = getattr(ds, "targets", None)
    if not targets:
        return test_loader
    if max_samples <= 0 or max_samples >= len(targets):
        return test_loader

    sub_indices = get_stratified_indices(targets, max_samples=max_samples, seed=seed)
    sub_ds = TransformSubset(ds, sub_indices, transform=None, raw_targets=targets)
    return DataLoader(
        sub_ds,
        batch_size=batch_size or test_loader.batch_size or 32,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_dataloaders(
    dataset_name: str = "intel",
    data_dir: Optional[str] = "./data",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_classes: Optional[int] = None,
    max_samples_per_class: Optional[int] = None,
    config_save_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train, validation, and test DataLoaders along with class names.

    Supported datasets:
        - 'intel': Intel Natural Scenes (6 classes, auto-downloaded via kagglehub).
        - 'sun397' / 'sun397_100': SUN397 Scene Dataset (full 397 or deterministic top-N subset).
        - 'custom': Local folder with 'train/' and 'test/' (and optional 'val/').
    """
    tfs = get_transforms(img_size)
    name = dataset_name.lower()

    if name == "intel":
        kaggle_input = Path("/kaggle/input/intel-image-classification")
        if kaggle_input.is_dir():
            downloaded_path = kaggle_input
        else:
            import kagglehub

            downloaded_path = Path(
                kagglehub.dataset_download("puneet6060/intel-image-classification")
            )
        train_dir, test_dir = _find_intel_split_dirs(downloaded_path)
        base_train = datasets.ImageFolder(root=str(train_dir))
        test_ds = datasets.ImageFolder(root=str(test_dir), transform=tfs["eval"])
        class_names = base_train.classes
        train_idx, val_idx = _stratified_split_indices(
            base_train.targets, val_ratio=val_ratio, seed=seed
        )
        train_ds = TransformSubset(
            base_train, train_idx, transform=tfs["train"], raw_targets=base_train.targets
        )
        val_ds = TransformSubset(
            base_train, val_idx, transform=tfs["eval"], raw_targets=base_train.targets
        )

    elif name in ("sun397", "sun397_100"):
        if name == "sun397_100" and max_classes is None:
            max_classes = 100
        root_path = Path(data_dir or "./data")
        base_ds = datasets.SUN397(root=str(root_path), download=True)
        raw_labels = np.asarray(base_ds._labels)
        all_classes = list(base_ds.classes)

        if max_classes is not None and max_classes < len(all_classes):
            # Deterministically select top-N classes by sample count (tie-broken by class index)
            unique_cls, counts = np.unique(raw_labels, return_counts=True)
            order = np.lexsort((unique_cls, -counts))
            chosen_orig_ids = sorted(unique_cls[order[:max_classes]].tolist())
        else:
            chosen_orig_ids = list(range(len(all_classes)))

        label_map = {orig_id: new_id for new_id, orig_id in enumerate(chosen_orig_ids)}
        class_names = [all_classes[i] for i in chosen_orig_ids]

        rng = np.random.default_rng(seed)
        filtered_indices: List[int] = []
        filtered_targets: List[int] = []
        for orig_id in chosen_orig_ids:
            cls_idx = np.where(raw_labels == orig_id)[0]
            rng.shuffle(cls_idx)
            if max_samples_per_class is not None:
                cls_idx = cls_idx[:max_samples_per_class]
            filtered_indices.extend(cls_idx.tolist())
            filtered_targets.extend([label_map[orig_id]] * len(cls_idx))

        train_rel, rest_rel = _stratified_split_indices(
            filtered_targets, val_ratio=0.30, seed=seed
        )
        rest_targets = [filtered_targets[i] for i in rest_rel]
        val_rel2, test_rel2 = _stratified_split_indices(
            rest_targets, val_ratio=0.50, seed=seed
        )

        train_idx = [filtered_indices[i] for i in train_rel]
        val_idx = [filtered_indices[rest_rel[i]] for i in val_rel2]
        test_idx = [filtered_indices[rest_rel[i]] for i in test_rel2]

        train_ds = TransformSubset(
            base_ds, train_idx, transform=tfs["train"], label_map=label_map, raw_targets=raw_labels
        )
        val_ds = TransformSubset(
            base_ds, val_idx, transform=tfs["eval"], label_map=label_map, raw_targets=raw_labels
        )
        test_ds = TransformSubset(
            base_ds, test_idx, transform=tfs["eval"], label_map=label_map, raw_targets=raw_labels
        )

    elif name == "custom":
        root_path = Path(data_dir or "./data")
        train_dir = root_path / "train"
        test_dir = root_path / "test"
        val_dir = root_path / "val"
        if val_dir.is_dir():
            train_ds = datasets.ImageFolder(root=str(train_dir), transform=tfs["train"])
            val_ds = datasets.ImageFolder(root=str(val_dir), transform=tfs["eval"])
            class_names = train_ds.classes
        else:
            base_train = datasets.ImageFolder(root=str(train_dir))
            class_names = base_train.classes
            train_idx, val_idx = _stratified_split_indices(
                base_train.targets, val_ratio=val_ratio, seed=seed
            )
            train_ds = TransformSubset(
                base_train, train_idx, transform=tfs["train"], raw_targets=base_train.targets
            )
            val_ds = TransformSubset(
                base_train, val_idx, transform=tfs["eval"], raw_targets=base_train.targets
            )
        test_ds = datasets.ImageFolder(root=str(test_dir), transform=tfs["eval"])

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose 'intel', 'sun397', 'sun397_100', or 'custom'."
        )

    if config_save_path:
        cfg_path = Path(config_save_path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_payload = {
            "dataset": dataset_name,
            "seed": seed,
            "num_classes": len(class_names),
            "class_names": class_names,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds),
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_payload, f, indent=2)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_names
