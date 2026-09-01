from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from apps.datasets.models import Dataset
from .models import TrainingRun


class TrainingConfigurationViewTests(TestCase):
    @patch("apps.training.views.run_training")
    def test_custom_configuration_is_saved_on_training_run(self, run_training):
        dataset = Dataset.objects.create(name="Ready dataset", status="ready")
        run_training.return_value = {
            "accuracy": 0.9,
            "macro_f1": 0.88,
            "epochs": 7,
            "split": "60 / 20 / 20",
            "split_ratios": [60, 20, 20],
        }

        response = self.client.post(reverse("training:start"), {
            "dataset": dataset.pk,
            "architecture": "efficientnet_b0",
            "train_split": "60",
            "validation_split": "20",
            "test_split": "20",
            "epochs": "7",
        })

        self.assertRedirects(response, reverse("training:home"))
        run = TrainingRun.objects.get(dataset=dataset)
        self.assertEqual(run.config["split_ratios"], [60, 20, 20])
        self.assertEqual(run.config["split"], "60 / 20 / 20")
        self.assertEqual(run.config["epochs"], 7)
        run_training.assert_called_once_with(run)

    def test_invalid_split_is_rejected_before_a_run_is_created(self):
        dataset = Dataset.objects.create(name="Ready dataset", status="ready")

        response = self.client.post(reverse("training:start"), {
            "dataset": dataset.pk,
            "architecture": "efficientnet_b0",
            "train_split": "70",
            "validation_split": "20",
            "test_split": "20",
            "epochs": "20",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "must be at least 1% and total 100%")
        self.assertFalse(dataset.training_runs.exists())

    def test_invalid_epoch_count_is_rejected_before_a_run_is_created(self):
        dataset = Dataset.objects.create(name="Ready dataset", status="ready")

        response = self.client.post(reverse("training:start"), {
            "dataset": dataset.pk,
            "architecture": "efficientnet_b0",
            "train_split": "70",
            "validation_split": "15",
            "test_split": "15",
            "epochs": "201",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Epochs must be between 1 and 200")
        self.assertFalse(dataset.training_runs.exists())

    def test_completed_run_can_be_selected_for_predictions(self):
        first = TrainingRun.objects.create(
            architecture="efficientnet_b0",
            status="completed",
            config={"checkpoint": "checkpoints/first.pt"},
            is_active=True,
        )
        second = TrainingRun.objects.create(
            architecture="mobilenet_v3",
            status="completed",
            config={"checkpoint": "checkpoints/second.pt"},
        )

        response = self.client.post(reverse("training:activate_model", args=[second.pk]))

        self.assertRedirects(response, reverse("training:experiments"))
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertFalse(first.is_active)
        self.assertTrue(second.is_active)

    def test_incomplete_run_cannot_be_selected_for_predictions(self):
        run = TrainingRun.objects.create(
            architecture="efficientnet_b0",
            status="running",
            config={},
        )

        response = self.client.post(reverse("training:activate_model", args=[run.pk]))

        self.assertRedirects(response, reverse("training:experiments"))
        run.refresh_from_db()
        self.assertFalse(run.is_active)