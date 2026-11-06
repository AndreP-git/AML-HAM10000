import numpy as np
import torch
import torch.nn.functional as F

from .transforms import denormalize


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
def gradcam_heatmap(model, target_layer, input_tensor, target_class=None, device="cpu"):
    """Return a (H, W) heatmap in [0, 1] for one image.

    ``input_tensor`` is a single normalized CHW tensor.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.to(device).eval()
    batch = input_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=batch, targets=targets)[0]  # (H, W)
    return grayscale


def overlay_heatmap(input_tensor, heatmap, mean=None, std=None, alpha=0.5):
    """Blend a heatmap over the de-normalized image"""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    rgb = denormalize(input_tensor, mean, std).permute(1, 2, 0).cpu().numpy()
    return show_cam_on_image(rgb.astype(np.float32), heatmap, use_rgb=True)


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def shap_explain(model, background, samples, device="cpu"):
    """Return SHAP values for ``samples`` given a ``background`` batch.

    ``background`` defines the reference distribution;
    ``samples`` are the images to explain.
    """
    import shap

    model.to(device).eval()
    explainer = shap.GradientExplainer(model, background.to(device))
    shap_values = explainer.shap_values(samples.to(device))
    return shap_values


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------
def lime_explain(model, image_tensor, eval_transform, device="cpu",
                 top_labels=3, num_samples=1000, mean=None, std=None, seed=42):
    """Explain one image with LIME. Returns ``(explanation, rgb_image)``.

    LIME works on raw RGB images and calls a prediction function that must
    accept a batch of HWC uint8 images.
    """
    from lime import lime_image
    from PIL import Image

    model.to(device).eval()
    rgb = (denormalize(image_tensor, mean, std).permute(1, 2, 0).cpu().numpy() * 255)
    rgb = rgb.astype(np.uint8)

    def predict_fn(images):
        batch = torch.stack([eval_transform(Image.fromarray(im)) for im in images])
        with torch.no_grad():
            logits = model(batch.to(device))
            return F.softmax(logits, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    explanation = explainer.explain_instance(
        rgb, predict_fn, top_labels=top_labels, hide_color=0, num_samples=num_samples
    )
    return explanation, rgb
