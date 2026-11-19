"""Model factory and architectural utilities for CNN, ViT, and CLIP backbones."""

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn

from src.clip_model import CLIPSceneClassifier, clip_vit_reshape_transform

MODEL_REGISTRY: Dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "convnext_tiny": "convnext_tiny",
    "clip_zeroshot": "ViT-B-16",
    "clip_linear": "ViT-B-16",
}


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape Swin Transformer block activations to (B, C, H, W) for CAM methods.

    Handles both 4D (B, H, W, C) tensors from timm >= 0.9 and 3D (B, H*W, C) tensors.
    """
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        b, n, c = tensor.shape
        h = w = int(math.isqrt(n))
        if h * w != n:
            raise ValueError(f"Expected square token grid, got {n} tokens.")
        return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2)
    raise ValueError(f"Unexpected Swin activation shape: {tuple(tensor.shape)}")


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    class_names: Optional[List[str]] = None,
) -> nn.Module:
    """Instantiate a classification backbone (CNN, Swin, or CLIP ViT-B/16)."""
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    if key in ("clip_zeroshot", "clip_linear"):
        if not class_names:
            class_names = [f"class_{i}" for i in range(num_classes)]
        mode = "zeroshot" if key == "clip_zeroshot" else "linear"
        return CLIPSceneClassifier(class_names=class_names, mode=mode)

    timm_name = MODEL_REGISTRY[key]
    return timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def get_target_layers_and_reshape(
    model: nn.Module, model_name: str
) -> Tuple[List[nn.Module], Optional[Callable[[torch.Tensor], torch.Tensor]]]:
    """Return the CAM target layer(s) and optional reshape_transform for a backbone."""
    key = model_name.lower()
    if key == "resnet50":
        return [model.layer4[-1]], None
    if key == "efficientnet_b0":
        return [model.conv_head], None
    if key == "convnext_tiny":
        return [model.stages[-1].blocks[-1]], None
    if key == "swin_tiny":
        return [model.layers[-1].blocks[-1].norm1], swin_reshape_transform
    if key in ("clip_zeroshot", "clip_linear"):
        return [model.visual.transformer.resblocks[-1].ln_1], clip_vit_reshape_transform
    raise ValueError(f"No CAM target layer mapping defined for '{model_name}'.")


def get_model_stats(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> Dict[str, float]:
    """Compute parameter counts (millions) and mean single-image inference latency (ms)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / timed_runs

    return {
        "total_params_m": round(total_params / 1e6, 2),
        "trainable_params_m": round(trainable_params / 1e6, 2),
        "latency_ms": round(elapsed * 1000.0, 2),
    }
