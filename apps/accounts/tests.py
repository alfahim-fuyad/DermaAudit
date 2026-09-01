import tempfile
from pathlib import Path

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.datasets.models import Dataset
from apps.training.models import TrainingRun
from .models import Profile


class AuthenticationFlowTests(TestCase):
    def test_registration_creates_user_profile_and_logs_in(self):
        response = self.client.post(reverse("accounts:register"), {
            "name": "Dr. Alex Morgan",
            "email": "Alex@Example.org",
            "organization": "Research Lab",
            "password": "safe-password-123",
        })

        self.assertRedirects(response, reverse("accounts:home"))
        user = User.objects.get(username="alex@example.org")
        self.assertTrue(user.check_password("safe-password-123"))
        self.assertEqual(user.profile.organization, "Research Lab")
        self.assertEqual(self.client.session.get("_auth_user_id"), str(user.pk))

    def test_registration_rejects_short_password(self):
        response = self.client.post(reverse("accounts:register"), {
            "name": "Alex Morgan",
            "email": "alex@example.org",
            "password": "short",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Password must be at least 8 characters.")
        self.assertFalse(User.objects.filter(username="alex@example.org").exists())

    def test_duplicate_email_is_rejected_case_insensitively(self):
        User.objects.create_user(username="alex@example.org", email="alex@example.org", password="safe-password-123")

        response = self.client.post(reverse("accounts:register"), {
            "name": "Another Alex",
            "email": "ALEX@EXAMPLE.ORG",
            "password": "safe-password-123",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "An account with that email already exists.")
        self.assertEqual(User.objects.filter(username="alex@example.org").count(), 1)

    def test_login_accepts_normalized_email(self):
        user = User.objects.create_user(
            username="alex@example.org",
            email="alex@example.org",
            password="safe-password-123",
        )
        Profile.objects.create(user=user)

        response = self.client.post(reverse("accounts:login"), {
            "username": " ALEX@EXAMPLE.ORG ",
            "password": "safe-password-123",
        })

        self.assertRedirects(response, reverse("accounts:home"))
        self.assertEqual(self.client.session.get("_auth_user_id"), str(user.pk))

    def test_invalid_login_shows_error_without_authenticating(self):
        User.objects.create_user(username="alex@example.org", password="safe-password-123")

        response = self.client.post(reverse("accounts:login"), {
            "username": "alex@example.org",
            "password": "wrong-password",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "That email or password was not recognised.")
        self.assertNotIn("_auth_user_id", self.client.session)


class OverviewRadarTests(TestCase):
    def test_radar_uses_live_dataset_counts_and_labels(self):
        Dataset.objects.create(
            name="Dermatology screening set",
            sample_count=24,
            classes=["melanoma", "nevus", "benign"],
        )
        Dataset.objects.create(
            name="Field crop health set",
            sample_count=10,
            classes=["healthy leaf", "rust"],
        )
        Dataset.objects.create(
            name="Object benchmark",
            sample_count=8,
            classes=["cup", "book"],
        )

        response = self.client.get(reverse("accounts:home"))
        radar = response.context["radar_data"]

        self.assertEqual(radar["total_dataset_count"], 3)
        self.assertEqual(radar["total_sample_count"], 42)
        self.assertEqual(radar["total_class_count"], 7)
        self.assertEqual(radar["primary_domain"], "health")
        self.assertEqual(radar["domains"]["health"]["dataset_count"], 1)
        self.assertEqual(radar["domains"]["health"]["class_count"], 3)
        self.assertEqual(radar["domains"]["environment"]["dataset_count"], 1)
        self.assertEqual(radar["domains"]["vision"]["dataset_count"], 1)

    def test_radar_status_endpoint_returns_fresh_json(self):
        Dataset.objects.create(
            name="Skin lesion images",
            sample_count=5,
            classes=["benign", "malignant"],
        )

        response = self.client.get(reverse("accounts:radar_status"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["domains"]["health"]["dataset_count"], 1)
        self.assertEqual(response.json()["domains"]["health"]["sample_count"], 5)

    def test_overview_status_returns_live_workspace_signals(self):
        dataset = Dataset.objects.create(
            name="Ready skin set",
            status="ready",
            sample_count=18,
            classes=["benign", "malignant"],
        )
        TrainingRun.objects.create(
            dataset=dataset,
            architecture="efficientnet_b0",
            status="running",
            config={"progress": {"percent": 42, "label": "Training the model"}},
        )

        response = self.client.get(reverse("accounts:overview_status"))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ready_dataset_count"], 1)
        self.assertEqual(payload["active_training_count"], 1)
        self.assertEqual(payload["active_runs"][0]["percent"], 42)
        self.assertEqual(payload["activity"][0]["kind"], "Training")

    def test_model_registry_shows_live_completed_checkpoint_data(self):
        dataset = Dataset.objects.create(name="Animals benchmark", status="ready")
        with tempfile.TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root):
            checkpoint = Path(media_root) / "checkpoints" / "training_run_1.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            run = TrainingRun.objects.create(
                dataset=dataset,
                architecture="mobilenet_v3",
                status="completed",
                accuracy=0.875,
                macro_f1=0.812,
                config={
                    "checkpoint": "checkpoints/training_run_1.pt",
                    "classes": ["cat", "dog", "horse"],
                    "epochs": 5,
                    "preprocessing": {"image_size": 64},
                },
            )

            response = self.client.get(reverse("accounts:home"))

        self.assertEqual(response.context["active_model"], run)
        self.assertEqual(response.context["model_registry"]["accuracy_percent"], 87.5)
        self.assertEqual(response.context["model_registry"]["class_count"], 3)
        self.assertContains(response, "MobileNetV3")
        self.assertContains(response, "87.5%")
        self.assertContains(response, "Animals benchmark")
        self.assertContains(response, "Use for prediction")

    def test_model_registry_stays_empty_without_a_valid_checkpoint(self):
        TrainingRun.objects.create(
            architecture="efficientnet_b0",
            status="completed",
            accuracy=0.99,
            config={"checkpoint": "checkpoints/missing.pt"},
        )

        response = self.client.get(reverse("accounts:home"))

        self.assertIsNone(response.context["active_model"])
        self.assertIsNone(response.context["model_registry"])
        self.assertContains(response, "No model registered yet")