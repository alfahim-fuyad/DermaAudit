import math
from pathlib import Path

from django.conf import settings
from PIL import Image

from apps.training.models import TrainingRun
from apps.training.services.trainer import _build_model, _image_tensor
from .confidence import decide_confidence
from .xai import generate_gradcam


MAX_PREDICTION_BYTES = 10 * 1024 * 1024
PREDICTION_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def validate_prediction_upload(uploaded_file):
    """Validate a single image before looking up a model or saving the file."""
    if not uploaded_file:
        raise ValueError("Choose an image before continuing.")
    if getattr(uploaded_file, "size", 0) > MAX_PREDICTION_BYTES:
        raise ValueError("The image exceeds the 10 MB upload limit.")
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()
    if suffix not in PREDICTION_EXTENSIONS:
        raise ValueError("Use a JPG, PNG, WEBP, BMP, GIF, or TIFF image.")
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            image.verify()
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            if not image.size[0] or not image.size[1]:
                raise ValueError("The image has no usable dimensions.")
    except (OSError, ValueError) as exc:
        raise ValueError(f"That file is not a readable image: {exc}") from exc
    finally:
        try:
            uploaded_file.seek(0)
        except (AttributeError, OSError):
            pass


def _is_within_media(media_root, candidate):
    try:
        try:
            return candidate.is_relative_to(media_root)
        except AttributeError:
            return media_root in candidate.parents or candidate == media_root
    except (ValueError, RuntimeError):
        return False


def _latest_checkpoint():
    media_root = Path(settings.MEDIA_ROOT).resolve()
    completed_runs = TrainingRun.objects.filter(
        status="completed",
        config__checkpoint__isnull=False,
    ).select_related("dataset")
    active_runs = completed_runs.filter(is_active=True).order_by("-created_at")
    runs = list(active_runs) + list(completed_runs.exclude(is_active=True).order_by("-created_at"))
    for run in runs:
        checkpoint_name = run.config.get("checkpoint")
        if not checkpoint_name:
            continue
        try:
            checkpoint_path = (media_root / checkpoint_name).resolve()
        except (ValueError, RuntimeError, OSError):
            continue
        if not _is_within_media(media_root, checkpoint_path) or not checkpoint_path.is_file():
            continue
        return run, checkpoint_path
    return None, None


def _analyze_with_checkpoint(uploaded_file):
    run, checkpoint_path = _latest_checkpoint()
    if not run:
        return None

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use a trained checkpoint.") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes") or []
    preprocessing = checkpoint.get("preprocessing") or {}
    image_size = int(preprocessing.get("image_size", checkpoint.get("image_size", 32)))
    if not classes or not checkpoint.get("state_dict"):
        raise ValueError("The selected training checkpoint is missing its model metadata.")

    model_version = int(checkpoint.get("model_version", 1))
    architecture = checkpoint.get("architecture")
    valid_architectures = {value for value, _label in TrainingRun.ARCHITECTURES}
    if architecture not in valid_architectures:
        raise ValueError("The selected training checkpoint has an unsupported model architecture.")
    class_to_index = checkpoint.get("class_to_index")
    if class_to_index:
        expected_mapping = {label: index for index, label in enumerate(classes)}
        if class_to_index != expected_mapping:
            raise ValueError("The selected training checkpoint has inconsistent class metadata.")
    if run.dataset and run.dataset.classes and sorted(run.dataset.classes) != sorted(classes):
        raise ValueError("The selected model is incompatible with its dataset class configuration.")
    model = _build_model(
        torch,
        nn,
        architecture,
        len(classes),
        model_version=model_version,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    uploaded_file.seek(0)
    with Image.open(uploaded_file) as image:
        tensor = _image_tensor(
            torch,
            image,
            image_size,
            normalize=bool(preprocessing.get("normalize", model_version >= 3)),
            mean=tuple(preprocessing.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(preprocessing.get("std", (0.229, 0.224, 0.225))),
        )
    tensor = tensor.unsqueeze(0)

    temperature = float(checkpoint.get("temperature", 1.0))
    if not math.isfinite(temperature):
        temperature = 1.0
    temperature = max(0.5, min(3.0, temperature))
    with torch.no_grad():
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[1] != len(classes):
            raise ValueError("The selected training checkpoint output does not match its class metadata.")
        probabilities = torch.softmax(logits / temperature, dim=1)[0]
    confidence, predicted_index = torch.max(probabilities, dim=0)
    predicted_index = int(predicted_index.item())
    if predicted_index >= len(classes):
        raise ValueError("The selected training checkpoint returned an unknown class.")

    ranked_predictions = [
        {
            "label": classes[index],
            "probability": round(float(probability.item()), 4),
            "percent": round(float(probability.item()) * 100, 2),
        }
        for probability, index in sorted(
            zip(probabilities, range(len(classes))),
            key=lambda item: float(item[0].item()),
            reverse=True,
        )[:3]
    ]
    confidence_value = float(confidence.item())
    confidence_policy = decide_confidence(confidence_value, ranked_predictions)
    xai_path = None
    uploaded_file.seek(0)
    with Image.open(uploaded_file) as original_image:
        xai_input = tensor.detach().clone().requires_grad_(True)
        xai_path = generate_gradcam(
            model,
            xai_input,
            predicted_index,
            original_image.copy(),
        )
    return {
        "label": classes[predicted_index],
        "confidence": round(confidence_value, 4),
        "confidence_percent": round(confidence_value * 100, 2),
        "ranked_predictions": ranked_predictions,
        "review_required": confidence_policy["review_required"],
        "decision": confidence_policy["decision"],
        "confidence_policy": confidence_policy,
        "mode": "trained_checkpoint",
        "model_run_id": run.pk,
        "model_architecture": run.get_architecture_display(),
        "model_dataset": run.dataset.name if run.dataset else "Unassigned dataset",
        "dataset_id": run.dataset_id,
        "xai_path": xai_path,
        "xai_url": f"{settings.MEDIA_URL}{xai_path}" if xai_path else None,
        "xai_status": "completed" if xai_path else "unavailable",
        "note": (
            f"Result generated by the completed {run.get_architecture_display()} "
            "checkpoint. Per-image confidence is separate from aggregate test accuracy. "
            "This is a model-assisted classification result and expert review remains required."
        ),
        "signals": [
            "Image format validated",
            f"{run.get_architecture_display()} checkpoint applied",
            "Human review recommended",
        ],
    }


def analyze_image(uploaded_file):
    """Use a valid selected checkpoint, then the newest valid completed checkpoint."""
    validate_prediction_upload(uploaded_file)
    checkpoint_analysis = _analyze_with_checkpoint(uploaded_file)
    if checkpoint_analysis is not None:
        return checkpoint_analysis
    raise RuntimeError(
        "No completed trained model is available. Complete an experiment and select "
        "a checkpoint before requesting a prediction."
    )
