"""Gradient-based visual explanation (Grad-CAM, Grad-CAM++, LayerCAM) and faithfulness metrics."""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader

from src.models import get_target_layers_and_reshape

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "layercam": LayerCAM,
}


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compute trapezoidal integral compatible with NumPy 1.x and 2.x."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def generate_cam_explanation(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    method: str = "gradcam",
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Generate a gradient-based CAM heatmap and overlay for a single image.

    Args:
        model: Trained PyTorch classification model.
        model_name: Key in MODEL_REGISTRY (e.g. 'resnet50', 'swin_tiny').
        input_tensor: Normalized tensor of shape (1, 3, H, W) on model's device.
        rgb_image: Float32 RGB image in [0, 1] of shape (H, W, 3).
        method: One of 'gradcam', 'gradcam++', 'layercam'.
        target_class: Class index to explain; defaults to top predicted class.

    Returns:
        (grayscale_cam, overlay_rgb, target_idx, target_probability)
    """
    method_key = method.lower()
    if method_key not in CAM_METHODS:
        raise ValueError(
            f"Unsupported XAI method '{method}'. Choose from {list(CAM_METHODS.keys())}."
        )

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        target_prob = float(probs[0, target_class].item())

    target_layers, reshape_tf = get_target_layers_and_reshape(model, model_name)
    cam_cls = CAM_METHODS[method_key]
    targets = [ClassifierOutputTarget(target_class)]

    with cam_cls(
        model=model, target_layers=target_layers, reshape_transform=reshape_tf
    ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return grayscale_cam, overlay, target_class, target_prob


@torch.no_grad()
def compute_insertion_deletion_auc(
    model: nn.Module,
    input_tensor: torch.Tensor,
    grayscale_cam: np.ndarray,
    target_class: int,
    steps: int = 20,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute Insertion AUC (higher=better) and Deletion AUC (lower=better).

    Uses a heavily blurred baseline image instead of hard zeros to minimize
    out-of-distribution edge artifacts during progressive pixel perturbation.
    """
    model.eval()
    device = input_tensor.device
    _, c, h, w = input_tensor.shape
    total_pixels = h * w

    # Smooth baseline via strong down-up bilinear blur
    blurred = F.avg_pool2d(input_tensor, kernel_size=21, stride=1, padding=10)
    flat_cam = grayscale_cam.reshape(-1)
    sorted_idx = np.argsort(-flat_cam)  # descending importance

    fractions = np.linspace(0.0, 1.0, steps + 1)
    ins_scores = []
    del_scores = []

    ins_batch = []
    del_batch = []

    for frac in fractions:
        k = int(round(frac * total_pixels))
        mask = np.zeros(total_pixels, dtype=np.float32)
        if k > 0:
            mask[sorted_idx[:k]] = 1.0
        mask_t = torch.from_numpy(mask.reshape(1, 1, h, w)).to(device)

        ins_img = input_tensor * mask_t + blurred * (1.0 - mask_t)
        del_img = input_tensor * (1.0 - mask_t) + blurred * mask_t
        ins_batch.append(ins_img)
        del_batch.append(del_img)

    ins_tensor = torch.cat(ins_batch, dim=0)
    del_tensor = torch.cat(del_batch, dim=0)

    ins_probs = torch.softmax(model(ins_tensor), dim=1)[:, target_class].cpu().numpy()
    del_probs = torch.softmax(model(del_tensor), dim=1)[:, target_class].cpu().numpy()

    ins_auc = _trapz(ins_probs, fractions)
    del_auc = _trapz(del_probs, fractions)
    return ins_auc, del_auc, ins_probs, del_probs


def evaluate_xai_faithfulness(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    methods: Iterable[str] = ("gradcam", "gradcam++", "layercam"),
    max_samples: int = 40,
    steps: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Benchmark XAI methods quantitatively using mean Insertion and Deletion AUC."""
    model.eval()
    model.to(device)
    methods_list = list(methods)
    scores: Dict[str, Dict[str, List[float]]] = {
        m: {"insertion_auc": [], "deletion_auc": []} for m in methods_list
    }

    processed = 0
    for images, _ in loader:
        for i in range(images.size(0)):
            if processed >= max_samples:
                break
            x = images[i : i + 1].to(device)
            with torch.no_grad():
                pred_cls = int(model(x).argmax(dim=1).item())

            # Dummy rgb not needed for AUC; pass zeros to reuse helper cleanly
            dummy_rgb = np.zeros((x.shape[2], x.shape[3], 3), dtype=np.float32)
            for m in methods_list:
                gcam, _, _, _ = generate_cam_explanation(
                    model, model_name, x, dummy_rgb, method=m, target_class=pred_cls
                )
                ins_auc, del_auc, _, _ = compute_insertion_deletion_auc(
                    model, x, gcam, target_class=pred_cls, steps=steps
                )
                scores[m]["insertion_auc"].append(ins_auc)
                scores[m]["deletion_auc"].append(del_auc)

            processed += 1
        if processed >= max_samples:
            break

    summary: Dict[str, Dict[str, float]] = {}
    for m in methods_list:
        summary[m] = {
            "insertion_auc": round(float(np.mean(scores[m]["insertion_auc"])), 4),
            "deletion_auc": round(float(np.mean(scores[m]["deletion_auc"])), 4),
        }
    return summary
