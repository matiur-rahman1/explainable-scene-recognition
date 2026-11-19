"""Reproducibility, seeding, and class-stratified sampling utilities."""

import os
import random
from typing import List, Sequence

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Lock random seeds across Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_stratified_indices(
    targets: Sequence[int],
    max_samples: int,
    seed: int = 42,
) -> List[int]:
    """Select up to max_samples indices uniformly balanced across all classes."""
    n_total = len(targets)
    if max_samples <= 0 or max_samples >= n_total:
        return list(range(n_total))

    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    classes = np.unique(targets_arr)
    n_classes = len(classes)
    per_class = max(1, max_samples // n_classes)
    remainder = max(0, max_samples - per_class * n_classes)

    selected: List[int] = []
    leftover: List[int] = []

    for i, cls_id in enumerate(classes):
        cls_idx = np.where(targets_arr == cls_id)[0]
        rng.shuffle(cls_idx)
        take = min(len(cls_idx), per_class + (1 if i < remainder else 0))
        selected.extend(cls_idx[:take].tolist())
        leftover.extend(cls_idx[take:].tolist())

    if len(selected) < max_samples and leftover:
        rng.shuffle(leftover)
        needed = max_samples - len(selected)
        selected.extend(leftover[:needed])

    rng.shuffle(selected)
    return selected[:max_samples]
