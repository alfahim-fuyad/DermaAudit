import os
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.datasets.models import Dataset
from apps.prediction.models import Prediction
from apps.training.models import TrainingRun


class ReportsDashboardTests(TestCase):
    def test_empty_dashboard_has_actionable_empty_states(self):
        response = self.client.get(reverse("reports:dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["audit_coverage"], 0)
        self.assertContains(response, "No completed experiments yet")
        self.assertContains(response, "Upload a dataset to start reliability reporting.")

    def test_dashboard_uses_real_run_metrics_and_audit_coverage(self):
        Dataset.objects.create(
            name="Ready dataset",
            status="ready",
            sample_count=120,
            class_count=3,
            audit={"quality_score": 92, "stages": {"audit": "completed"}},
        )
        TrainingRun.objects.create(
            architecture="mobilenet_v3",
            status="completed",
            accuracy=0.9,
            macro_f1=0.84,
        )

        response = self.client.get(reverse("reports:dashboard"))

        self.assertEqual(response.context["audit_coverage"], 100)
        self.assertEqual(response.context["best_accuracy"], 90.0)
        self.assertEqual(response.context["performance_chart"][0]["accuracy"], 90.0)
        self.assertContains(response, "90.0%")
        self.assertContains(response, "MobileNetV3")

    def test_clear_all_removes_records_and_saved_media(self):
        with tempfile.TemporaryDirectory() as media_root:
            with override_settings(MEDIA_ROOT=media_root):
                dataset = Dataset.objects.create(
                    name="Old dataset",
                    uploaded_file=SimpleUploadedFile("old.zip", b"archive"),
                )
                run = TrainingRun.objects.create(
                    dataset=dataset,
                    config={"checkpoint": "checkpoints/old.pt"},
                )
                checkpoint_path = os.path.join(media_root, "checkpoints", "old.pt")
                os.makedirs(os.path.dirname(checkpoint_path))
                with open(checkpoint_path, "wb") as checkpoint:
                    checkpoint.write(b"weights")
                prediction = Prediction.objects.create(
                    image=SimpleUploadedFile("old.png", b"image"),
                    predicted_class="old",
                )
                dataset_path = dataset.uploaded_file.path
                prediction_path = prediction.image.path

                response = self.client.post(reverse("reports:clear_all"))

                self.assertRedirects(response, reverse("reports:dashboard"))
                self.assertFalse(Dataset.objects.exists())
                self.assertFalse(TrainingRun.objects.exists())
                self.assertFalse(Prediction.objects.exists())
                self.assertFalse(os.path.exists(dataset_path))
                self.assertFalse(os.path.exists(checkpoint_path))
                self.assertFalse(os.path.exists(prediction_path))

    def test_clear_endpoints_only_accept_post(self):
        self.assertEqual(self.client.get(reverse("datasets:clear")).status_code, 405)
        self.assertEqual(self.client.get(reverse("training:clear_experiments")).status_code, 405)
        self.assertEqual(self.client.get(reverse("prediction:clear_history")).status_code, 405)
        self.assertEqual(self.client.get(reverse("reports:clear_all")).status_code, 405)

    def test_live_status_returns_current_workspace_summary(self):
        Dataset.objects.create(name="Ready dataset", status="ready")
        TrainingRun.objects.create(
            architecture="efficientnet_b0",
            status="running",
            config={"progress": {"percent": 42, "label": "Training the model"}},
        )

        response = self.client.get(reverse("reports:live_status"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["datasets"], 1)
        self.assertEqual(response.json()["active_runs"][0]["progress"]["percent"], 42)


class AuditReportTests(TestCase):
    def test_audit_report_renders_the_shared_two_column_grid_wrapper(self):
        dataset = Dataset.objects.create(
            name="Audited dataset",
            status="ready",
            sample_count=24,
            class_count=2,
            validation={"readable_count": 24},
            audit={"quality_score": 94, "recommendation": "Ready for training."},
            pipeline={"stages": {"validation": "completed", "audit": "completed"}},
        )

        response = self.client.get(reverse("reports:audit_report", args=[dataset.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="audit-report-page"')
        self.assertContains(response, 'class="report-kpis"')
        self.assertContains(response, "Ready for training.")