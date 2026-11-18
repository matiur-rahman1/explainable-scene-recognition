"""Dataset loading, preprocessing, and normalization utilities for scene recognition."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransformSubset(Dataset):
    """Wraps a dataset subset with an independent transform pipeline."""

    def __init__(self, dataset: Dataset, indices: List[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
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
            # Verify they contain class subdirectories rather than another nested split folder
            subdirs = [p for p in train_dir.iterdir() if p.is_dir()]
            if subdirs and subdirs[0].name != "seg_train":
                return train_dir, test_dir
    raise FileNotFoundError(
        f"Could not locate seg_train / seg_test class folders under {root}"
    )


def _stratified_split_indices(
    targets: List[int], val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split sample indices into stratified train and validation sets."""
    rng = np.random.default_rng(seed)
    targets_arr = np.array(targets)
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


def create_dataloaders(
    dataset_name: str = "intel",
    data_dir: Optional[str] = "./data",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train, validation, and test DataLoaders along with class names.

    Supported datasets:
        - 'intel': Intel Natural Scenes (6 classes, auto-downloaded via kagglehub).
        - 'sun397': SUN397 Scene Dataset (397 classes, via torchvision).
        - 'custom': Local folder with 'train/' and 'test/' (and optional 'val/').
    """
    tfs = get_transforms(img_size)
    name = dataset_name.lower()

    if name == "intel":
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
        train_ds = TransformSubset(base_train, train_idx, transform=tfs["train"])
        val_ds = TransformSubset(base_train, val_idx, transform=tfs["eval"])

    elif name == "sun397":
        root_path = Path(data_dir or "./data")
        base_ds = datasets.SUN397(root=str(root_path), download=True)
        class_names = base_ds.classes
        train_idx, rest_idx = _stratified_split_indices(
            base_ds._labels, val_ratio=0.30, seed=seed
        )
        rest_labels = [base_ds._labels[i] for i in rest_idx]
        val_rel, test_rel = _stratified_split_indices(
            rest_labels, val_ratio=0.50, seed=seed
        )
        val_idx = [rest_idx[i] for i in val_rel]
        test_idx = [rest_idx[i] for i in test_rel]
        train_ds = TransformSubset(base_ds, train_idx, transform=tfs["train"])
        val_ds = TransformSubset(base_ds, val_idx, transform=tfs["eval"])
        test_ds = TransformSubset(base_ds, test_idx, transform=tfs["eval"])

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
            train_ds = TransformSubset(base_train, train_idx, transform=tfs["train"])
            val_ds = TransformSubset(base_train, val_idx, transform=tfs["eval"])
        test_ds = datasets.ImageFolder(root=str(test_dir), transform=tfs["eval"])

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose 'intel', 'sun397', or 'custom'."
        )

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
