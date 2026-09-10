"""Grad-CAM++ implementation for improved localization.

This builds on the base Grad-CAM in ml/xai/gradcam.py and apps/prediction/services/xai.py,
providing a slightly more precise heatmap using second-order gradients.
Falls back gracefully when torch is unavailable.
"""
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from PIL import Image


def _heat_color(value: float):
    value = max(0.0, min(1.0, float(value)))
    return (
        int(255 * value),
        int(255 * min(1.0, value * 2)),
        int(255 * (1.0 - value)),
    )


def generate_gradcam_plus(model, input_tensor, target_index, original_image):
    """Generate Grad-CAM++ overlay and persist it.

    Returns relative media path or None if unavailable.
    """
    try:
        import torch
        from torch import nn
    except ImportError:
        return None

    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        return None

    activations = {}
    gradients = {}

    target_layer = conv_layers[-1]

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        if hasattr(output, "register_hook"):
            output.register_hook(lambda grad: gradients.setdefault("value", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        out = model(input_tensor)
        if out.ndim != 2 or target_index >= out.shape[1]:
            return None
        # First-order
        score = out[:, target_index].sum()
        score.backward(retain_graph=True)

        act = activations.get("value")
        grad = gradients.get("value")
        if act is None or grad is None or act.ndim != 4:
            return None

        # Grad-CAM++: use squared grads and extra weighting
        grad2 = grad * grad
        grad3 = grad2 * grad

        # alpha = grad2 / (2*grad2 + sum(act * grad3))
        denom = 2 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grad2 / denom

        # weights = sum(alpha * relu(grad))
        weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)

        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=original_image.size[::-1], mode="bilinear", align_corners=False
        )[0, 0].detach()

        max_val = float(cam.max().item())
        if max_val <= 0:
            return None
        cam = cam / max_val

        heatmap = Image.new("RGB", original_image.size)
        px = heatmap.load()
        vals = cam.cpu().tolist()
        for y, row in enumerate(vals):
            for x, v in enumerate(row):
                px[x, y] = _heat_color(v)

        overlay = Image.blend(original_image.convert("RGB"), heatmap, 0.42)
        rel_path = f"xai/{uuid4().hex}_plus.png"
        dest = Path(settings.MEDIA_ROOT) / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(dest, format="PNG")
        return rel_path
    except (RuntimeError, ValueError, OSError):
        return None
    finally:
        handle.remove()
