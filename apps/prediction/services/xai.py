"""Grad-CAM artifact generation for the small CNN checkpoints used by training."""
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from PIL import Image


def _heat_color(value):
    """Map a normalized CAM value to a simple blue-to-red heat color."""
    value = max(0.0, min(1.0, float(value)))
    return (
        int(255 * value),
        int(255 * min(1.0, value * 2)),
        int(255 * (1.0 - value)),
    )


def generate_gradcam(model, input_tensor, target_index, original_image):
    """Persist a Grad-CAM overlay and return its workspace-relative media path.

    The function intentionally returns None for unsupported/degenerate models so
    explainability never blocks the base prediction.
    """
    try:
        import torch
        from torch import nn
    except ImportError:
        return None

    convolution_layers = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    if not convolution_layers:
        return None

    activations = {}
    gradients = {}
    target_layer = convolution_layers[-1]

    def capture_activation(_module, _inputs, output):
        activations["value"] = output
        if hasattr(output, "register_hook"):
            output.register_hook(lambda gradient: gradients.setdefault("value", gradient))

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        model_output = model(input_tensor)
        if model_output.ndim != 2 or target_index >= model_output.shape[1]:
            return None
        model_output[:, target_index].sum().backward()
        activation = activations.get("value")
        gradient = gradients.get("value")
        if activation is None or gradient is None or activation.ndim != 4:
            return None
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=original_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[0, 0].detach()
        maximum = float(cam.max().item())
        if maximum <= 0:
            return None
        cam = cam / maximum
        heatmap = Image.new("RGB", original_image.size)
        pixels = heatmap.load()
        values = cam.cpu().tolist()
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                pixels[x, y] = _heat_color(value)
        overlay = Image.blend(original_image.convert("RGB"), heatmap, 0.42)
        relative_path = f"xai/{uuid4().hex}.png"
        destination = Path(settings.MEDIA_ROOT) / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(destination, format="PNG")
        return relative_path
    except (RuntimeError, ValueError, OSError):
        return None
    finally:
        handle.remove()
