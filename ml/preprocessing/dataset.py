"""Dataset utilities for image-classification workflows.

This module provides a lightweight wrapper around the training service's
tensor dataset logic, so that external scripts or notebooks can reuse the
same preprocessing without duplicating code.
"""
import io
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

try:
    import torch
    from apps.training.services.trainer import IMAGE_SIZE, _image_tensor
except ImportError:  # pragma: no cover - torch optional for some environments
    torch = None
    IMAGE_SIZE = 64
    _image_tensor = None


def load_image_bytes(image_path: Path | str) -> bytes:
    """Read an image file as bytes."""
    path = Path(image_path)
    return path.read_bytes()


def image_bytes_to_tensor(image_bytes: bytes, image_size: int = IMAGE_SIZE):
    """Convert raw image bytes to a normalized CHW tensor."""
    if torch is None or _image_tensor is None:
        raise RuntimeError("PyTorch is required for tensor conversion.")
    with Image.open(io.BytesIO(image_bytes)) as image:
        return _image_tensor(torch, image, image_size)


def build_tensor_dataset(
    samples: Iterable[Tuple[bytes, str]],
    label_to_index: Dict[str, int],
    augment: bool = False,
    image_size: int = IMAGE_SIZE,
):
    """Build a TensorDataset from (image_bytes, label) pairs.

    Args:
        samples: Iterable of (image_bytes, label_string).
        label_to_index: Mapping from label string to integer index.
        augment: Whether to include a horizontal flip augmentation.
        image_size: Target image size for resizing.

    Returns:
        torch.utils.data.TensorDataset or None if no samples.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to build a tensor dataset.")

    # Reuse the trainer's internal logic when available to keep preprocessing identical
    try:
        from apps.training.services.trainer import _tensor_dataset

        return _tensor_dataset(torch, list(samples), label_to_index, augment=augment)
    except ImportError:
        # Fallback minimal implementation
        from PIL import Image

        images: List[torch.Tensor] = []
        labels: List[int] = []
        for image_bytes, label in samples:
            with Image.open(io.BytesIO(image_bytes)) as img:
                tensor = image_bytes_to_tensor(image_bytes, image_size=image_size)
            images.append(tensor)
            labels.append(label_to_index[label])
            if augment:
                images.append(torch.flip(tensor, dims=(2,)))
                labels.append(label_to_index[label])
        if not images:
            return None
        return torch.utils.data.TensorDataset(
            torch.stack(images), torch.tensor(labels)
        )
