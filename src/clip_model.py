"""CLIP (ViT-B/16) Zero-Shot and Linear-Probe wrappers compatible with CAM and PGD."""

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PROMPT_TEMPLATES: Sequence[str] = (
    "a photo of a {}.",
    "a photo of the {} scene.",
    "an image of a {}.",
    "a natural scene showing {}.",
    "a photograph of a {} environment.",
)


def clip_vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT-B/16 token activations (with CLS token) to (B, C, H, W) for CAM."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D ViT activation tensor, got {tuple(tensor.shape)}")
    s0, s1, c = tensor.shape
    h0 = int(math.isqrt(max(0, s0 - 1)))
    h1 = int(math.isqrt(max(0, s1 - 1)))
    if s1 > 1 and h1 * h1 == s1 - 1:
        # Shape is (B, 1 + H*W, C)
        patch_tokens = tensor[:, 1:, :]
        b = s0
        h = w = h1
    elif s0 > 1 and h0 * h0 == s0 - 1:
        # Shape is (1 + H*W, B, C)
        patch_tokens = tensor[1:, :, :].permute(1, 0, 2)
        b = s1
        h = w = h0
    else:
        raise ValueError(f"Cannot infer ViT token grid from shape {tuple(tensor.shape)}")
    return patch_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)


def _clean_class_label(raw_name: str) -> str:
    """Convert dataset folder names (e.g. '/a/abbey' or 'forest_path') into clean text."""
    cleaned = raw_name.strip("/").split("/")[-1].replace("_", " ")
    return cleaned


class CLIPSceneClassifier(nn.Module):
    """Wraps open_clip ViT-B/16 for Zero-Shot or Linear-Probe scene classification.

    Accepts standard ImageNet-normalized tensors (B, 3, 224, 224) and converts them
    to CLIP normalization internally so that DataLoaders, Grad-CAM, and FGSM/PGD
    operate uniformly across CNNs, Swin, and CLIP.
    """

    def __init__(
        self,
        class_names: List[str],
        mode: str = "zeroshot",
        model_arch: str = "ViT-B-16",
        pretrained: str = "openai",
    ):
        super().__init__()
        import open_clip

        self.mode = mode.lower()
        self.class_names = class_names
        self.num_classes = len(class_names)

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_arch, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_arch)
        self.visual = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp().detach()

        # Register normalization buffers for differentiable ImageNet -> CLIP re-normalization
        im_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        im_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        cl_mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        cl_std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("im_mean", im_mean)
        self.register_buffer("im_std", im_std)
        self.register_buffer("cl_mean", cl_mean)
        self.register_buffer("cl_std", cl_std)

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.visual(dummy).shape[-1]

        if self.mode == "zeroshot":
            for p in self.visual.parameters():
                p.requires_grad = False
            text_weights = self._build_zeroshot_weights(clip_model, tokenizer, class_names)
            self.register_buffer("text_Features", text_weights)
            self.head = None
        elif self.mode == "linear":
            for p in self.visual.parameters():
                p.requires_grad = False
            self.head = nn.Linear(feat_dim, self.num_classes)
            # Initialize linear probe with zero-shot prompt-ensembled text embeddings
            text_weights = self._build_zeroshot_weights(clip_model, tokenizer, class_names)
            with torch.no_grad():
                self.head.weight.copy_(text_weights.T * float(self.logit_scale.item()))
                self.head.bias.zero_()
        else:
            raise ValueError(f"Unknown CLIP mode '{mode}'. Use 'zeroshot' or 'linear'.")

    @torch.no_grad()
    def _build_zeroshot_weights(
        self, clip_model: nn.Module, tokenizer, class_names: List[str]
    ) -> torch.Tensor:
        """Compute prompt-ensembled L2-normalized text embeddings of shape (D, K)."""
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
