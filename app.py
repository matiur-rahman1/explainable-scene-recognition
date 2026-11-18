"""Interactive Gradio demo for Explainable Multi-Class Scene Recognition."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms
from src.models import MODEL_REGISTRY, build_model
from src.xai import compute_insertion_deletion_auc, generate_cam_explanation

DEFAULT_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, List[str]]] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EVAL_TF = get_transforms(224)["eval"]


def _load_model(model_name: str, checkpoint_dir: str = "./checkpoints"):
    """Load and cache a fine-tuned checkpoint if available, else pretrained backbone."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    ckpt_path = Path(checkpoint_dir) / f"best_{model_name}.pth"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
        class_names = ckpt.get("class_names", DEFAULT_CLASSES)
        model = build_model(model_name, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        class_names = DEFAULT_CLASSES
        model = build_model(model_name, num_classes=len(class_names), pretrained=True)

    model.eval().to(_DEVICE)
    _MODEL_CACHE[model_name] = (model, class_names)
    return model, class_names


def predict_and_explain(
    image: Image.Image,
    model_name: str,
    xai_method: str,
    checkpoint_dir: str = "./checkpoints",
):
    """Run scene classification, CAM explanation, and Insertion/Deletion AUC."""
    if image is None:
        return None, {}, "Please upload a scene image."

    rgb_pil = image.convert("RGB")
    resized_pil = rgb_pil.resize((224, 224))
    rgb_np = np.asarray(resized_pil, dtype=np.float32) / 255.0

    input_tensor = _EVAL_TF(rgb_pil).unsqueeze(0).to(_DEVICE)
    model, class_names = _load_model(model_name, checkpoint_dir=checkpoint_dir)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()

    confidences = {cls: float(p) for cls, p in zip(class_names, probs)}

    grayscale_cam, overlay, pred_idx, pred_prob = generate_cam_explanation(
        model=model,
        model_name=model_name,
        input_tensor=input_tensor,
        rgb_image=rgb_np,
        method=xai_method,
    )
    ins_auc, del_auc, _, _ = compute_insertion_deletion_auc(
        model=model,
        input_tensor=input_tensor,
        grayscale_cam=grayscale_cam,
        target_class=pred_idx,
        steps=16,
    )

    metrics_md = (
        f"### Explanation Faithfulness ({xai_method.upper()})\n"
        f"- **Predicted Scene:** `{class_names[pred_idx]}` ({pred_prob * 100:.2f}%)\n"
        f"- **Insertion AUC (↑ higher is better):** `{ins_auc:.4f}`\n"
        f"- **Deletion AUC (↓ lower is better):** `{del_auc:.4f}`"
    )
    return overlay, confidences, metrics_md


def build_interface(checkpoint_dir: str = "./checkpoints") -> gr.Blocks:
    """Create the Gradio Blocks UI."""
    with gr.Blocks(title="Explainable Scene Recognition (XAI)") as demo:
        gr.Markdown(
            "# Explainable Multi-Class Scene Recognition\n"
            "Compare **CNN (ResNet-50, EfficientNet-B0)** vs. **Vision Transformer (Swin-Tiny)** "
            "with **Grad-CAM / Grad-CAM++ / LayerCAM** and quantitative **Insertion/Deletion AUC**."
        )
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Upload Natural Scene Image")
                model_dd = gr.Dropdown(
                    choices=list(MODEL_REGISTRY.keys()),
                    value="swin_tiny",
                    label="Backbone Architecture",
                )
                xai_dd = gr.Dropdown(
                    choices=["gradcam", "gradcam++", "layercam"],
                    value="gradcam",
                    label="Gradient-Based XAI Method",
                )
                run_btn = gr.Button("Classify & Explain", variant="primary")

            with gr.Column():
                out_overlay = gr.Image(type="numpy", label="Visual Explanation Heatmap")
                out_label = gr.Label(num_top_classes=6, label="Scene Probabilities")
                out_metrics = gr.Markdown()

        run_btn.click(
            fn=lambda img, m, x: predict_and_explain(img, m, x, checkpoint_dir),
            inputs=[in_img, model_dd, xai_dd],
            outputs=[out_overlay, out_label, out_metrics],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_interface(checkpoint_dir=args.checkpoint_dir)
    app.launch(share=args.share)
