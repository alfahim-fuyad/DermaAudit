import io

from django.test import SimpleTestCase
from PIL import Image

from .services.trainer import (
    EPOCHS,
    IMAGE_SIZE,
    MAX_BATCH_SIZE,
    _build_model,
    _image_tensor,
    _split_records,
    _tensor_dataset,
)


class TrainingPipelineTests(SimpleTestCase):
    def test_quick_training_defaults_are_bounded(self):
        self.assertEqual(IMAGE_SIZE, 64)
        self.assertEqual(EPOCHS, 5)
        self.assertEqual(MAX_BATCH_SIZE, 32)

    def test_training_images_are_normalized_and_augmented(self):
        import torch

        image = Image.new("RGB", (80, 48), (128, 96, 64))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        dataset = _tensor_dataset(
            torch,
            [(buffer.getvalue(), "paper")],
            {"paper": 0},
            augment=True,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.tensors[0].shape, (2, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertFalse(torch.all((dataset.tensors[0] >= 0) & (dataset.tensors[0] <= 1)))

    def test_version_three_model_returns_class_logits(self):
        import torch
        from torch import nn

        model = _build_model(torch, nn, "efficientnet_b0", 3, model_version=3)
        output = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))

        self.assertEqual(tuple(output.shape), (1, 3))

    def test_custom_split_is_applied_per_class(self):
        records = [
            {"label": "paper", "filename": f"paper/{index}.png"}
            for index in range(10)
        ] + [
            {"label": "rock", "filename": f"rock/{index}.png"}
            for index in range(10)
        ]

        train, validation, test = _split_records(records, (60, 20, 20))

        self.assertEqual([len(part) for part in (train, validation, test)], [12, 4, 4])
        self.assertEqual({record["label"] for record in validation}, {"paper", "rock"})