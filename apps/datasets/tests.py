import io
import os
import tempfile
import zipfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from PIL import Image

from .services.validator import validate_upload
from .services.profiler import build_profile
from .services.leakage import analyze_leakage
from .services.bias import analyze_bias
from .services.auditor import build_audit
from .models import Dataset
from .services.pipeline import run_dataset_pipeline
from apps.training.models import TrainingRun


def _image_bytes(color="white", size=(64, 48)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def _zip_file(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for filename, content in entries:
            archive.writestr(filename, content)
    return SimpleUploadedFile(
        "dataset.zip",
        stream.getvalue(),
        content_type="application/zip",
    )


class DatasetValidatorTests(SimpleTestCase):
    def test_valid_two_class_archive_is_accepted_and_duplicates_are_counted(self):
        image = _image_bytes()
        report = validate_upload(_zip_file([
            ("benign/one.png", image),
            ("benign/duplicate.png", image),
            ("malignant/two.png", _image_bytes("black")),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["classes"], ["benign", "malignant"])
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["duplicate_count"], 1)

    def test_corrupt_image_is_rejected(self):
        report = validate_upload(_zip_file([
            ("benign/broken.png", b"not an image"),
            ("malignant/ok.png", _image_bytes()),
        ]))

        self.assertFalse(report["valid"])
        self.assertEqual(report["invalid_count"], 1)
        self.assertTrue(report["errors"])

    def test_flat_images_are_not_treated_as_a_classification_dataset(self):
        report = validate_upload(_zip_file([
            ("one.png", _image_bytes()),
            ("two.png", _image_bytes("black")),
        ]))

        self.assertFalse(report["valid"])
        self.assertIn("At least two labelled class folders are required.", report["errors"])

    def test_non_zip_upload_is_rejected(self):
        report = validate_upload(SimpleUploadedFile("dataset.tar", b"archive"))

        self.assertFalse(report["valid"])
        self.assertIn("Upload a ZIP containing one folder per class.", report["errors"])

    def test_path_traversal_is_rejected(self):
        report = validate_upload(_zip_file([
            ("../benign/escape.png", _image_bytes()),
            ("malignant/ok.png", _image_bytes("black")),
        ]))

        self.assertFalse(report["valid"])
        self.assertTrue(any("Unsafe archive path" in error for error in report["errors"]))

    def test_metadata_is_matched_to_images_and_enables_group_split(self):
        report = validate_upload(_zip_file([
            ("benign/one.png", _image_bytes()),
            ("malignant/two.png", _image_bytes("black")),
            (
                "metadata.csv",
                b"filename,label,patient_id,sex\n"
                b"one.png,benign,patient-1,F\n"
                b"two.png,malignant,patient-2,M\n",
            ),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["metadata"]["matched_images"], 2)
        self.assertEqual(report["metadata"]["group_column"], "patient_id")
        self.assertEqual(report["metadata"]["group_count"], 2)

        profile = build_profile(report)
        leakage = analyze_leakage(report)
        bias = analyze_bias(profile)

        self.assertTrue(leakage["group_information_available"])
        self.assertEqual(leakage["split_strategy"], "group-aware")
        self.assertTrue(bias["available"])
        self.assertIn("sex", bias["subgroups"])

    def test_metadata_label_mismatch_is_reported(self):
        report = validate_upload(_zip_file([
            ("benign/one.png", _image_bytes()),
            ("malignant/two.png", _image_bytes("black")),
            (
                "metadata.csv",
                b"filename,label\none.png,malignant\ntwo.png,malignant\n",
            ),
        ]))

        self.assertEqual(report["metadata"]["label_mismatch_count"], 1)
        profile = build_profile(report)
        audit = build_audit(
            report,
            profile,
            analyze_leakage(report),
            analyze_bias(profile),
            {"ratio": "1:1.0", "recommendation": "Class distribution is reasonably balanced."},
        )
        self.assertEqual(audit["label_consistency"], 50)
        self.assertIn("mismatches", audit["recommendation"])

    def test_low_resolution_images_are_flagged_without_being_discarded(self):
        report = validate_upload(_zip_file([
            ("benign/tiny.png", _image_bytes(size=(16, 16))),
            ("malignant/normal.png", _image_bytes("black")),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["low_quality_count"], 1)
        self.assertTrue(any("smaller than" in warning for warning in report["warnings"]))


class DatasetPipelineTests(TestCase):
    def test_pipeline_persists_public_audit_results(self):
        dataset = Dataset.objects.create(
            name="Pipeline test",
            uploaded_file=_zip_file([
                ("benign/one.png", _image_bytes()),
                ("malignant/two.png", _image_bytes("black")),
            ]),
        )

        result = run_dataset_pipeline(dataset)

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.class_count, 2)
        self.assertEqual(result.pipeline["status"], "completed")
        self.assertNotIn("records", result.validation)
        self.assertEqual(result.audit["leakage_risk"], "Low")


class DatasetAuditViewTests(TestCase):
    def test_completed_audit_marks_all_progress_steps_complete(self):
        dataset = Dataset.objects.create(
            name="Completed audit",
            status="ready",
            profile={"formats": "PNG"},
            audit={
                "quality_score": 94,
                "stages": {
                    "leakage": "completed",
                    "imbalance": "completed",
                },
            },
        )

        response = self.client.get(reverse("datasets:audit", args=[dataset.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["progress_percent"], 100)
        self.assertTrue(all(response.context["progress"].values()))
        self.assertContains(response, 'class="audit-progress is-complete"')
        self.assertEqual(response.context["current_step"], "ready")

    def test_unrun_audit_starts_at_profile_step(self):
        dataset = Dataset.objects.create(name="New audit", status="uploaded")

        response = self.client.get(reverse("datasets:audit", args=[dataset.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["progress_percent"], 0)
        self.assertEqual(response.context["current_step"], "profile")
        self.assertContains(response, 'class="audit-step active"')


class DatasetDeletionTests(TestCase):
    def test_delete_requires_post(self):
        dataset = Dataset.objects.create(
            name="Protected dataset",
            uploaded_file=SimpleUploadedFile("protected.zip", b"archive", content_type="application/zip"),
        )

        response = self.client.get(reverse("datasets:delete", args=[dataset.pk]))

        self.assertEqual(response.status_code, 405)
        self.assertTrue(Dataset.objects.filter(pk=dataset.pk).exists())

    def test_delete_removes_dataset_archive_and_training_runs(self):
        with tempfile.TemporaryDirectory() as media_root:
            with self.settings(MEDIA_ROOT=media_root):
                dataset = Dataset.objects.create(
                    name="Removable dataset",
                    uploaded_file=SimpleUploadedFile(
                        "removable.zip",
                        b"archive",
                        content_type="application/zip",
                    ),
                )
                TrainingRun.objects.create(dataset=dataset)
                stored_path = dataset.uploaded_file.path

                response = self.client.post(
                    reverse("datasets:delete", args=[dataset.pk]),
                    follow=True,
                )

                self.assertRedirects(response, reverse("datasets:list"))
                self.assertFalse(Dataset.objects.filter(pk=dataset.pk).exists())
                self.assertFalse(TrainingRun.objects.filter(dataset_id=dataset.pk).exists())
                self.assertFalse(os.path.exists(stored_path))
                self.assertContains(response, "Removable dataset was deleted from your workspace.")