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
from .services.imbalance import analyze_imbalance
from .services.auditor import build_audit
from .services.cleaner import build_cleaning_report, clean_records
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

    def test_corrupt_image_is_excluded_by_cleaning(self):
        report = validate_upload(_zip_file([
            ("benign/broken.png", b"not an image"),
            ("benign/ok.png", _image_bytes()),
            ("benign/ok_two.png", _image_bytes("gray")),
            ("malignant/one.png", _image_bytes("black")),
            ("malignant/two.png", _image_bytes("red")),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["invalid_count"], 1)
        self.assertEqual(report["invalid_files"], ["benign/broken.png"])
        self.assertTrue(
            any("excluded by automatic cleaning" in warning for warning in report["warnings"])
        )

    def test_heavily_damaged_archive_needs_review(self):
        entries = [("malignant/ok.png", _image_bytes("black"))]
        for index in range(3):
            entries.append((f"benign/broken_{index}.png", b"not an image"))

        report = validate_upload(_zip_file(entries))

        self.assertFalse(report["valid"])
        self.assertTrue(any("looks damaged" in error for error in report["errors"]))
        self.assertTrue(report["errors"])

    def test_flat_images_are_not_treated_as_a_classification_dataset(self):
        report = validate_upload(_zip_file([
            ("one.png", _image_bytes()),
            ("two.png", _image_bytes("black")),
        ]))

        self.assertFalse(report["valid"])
        self.assertIn("At least two labelled classes are required.", report["errors"])

    def test_flat_images_with_metadata_labels_are_accepted(self):
        report = validate_upload(_zip_file([
            ("one.bmp", _image_bytes()),
            ("two.tiff", _image_bytes("black")),
            (
                "labels.csv",
                b"filename,category\none.bmp,cats\ntwo.tiff,dogs\n",
            ),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["classes"], ["cats", "dogs"])
        self.assertEqual(report["class_counts"], {"cats": 1, "dogs": 1})
        self.assertEqual(report["metadata"]["matched_images"], 2)

    def test_images_folder_with_metadata_labels_uses_manifest_classes(self):
        report = validate_upload(_zip_file([
            ("images/one.bmp", _image_bytes()),
            ("images/two.tiff", _image_bytes("black")),
            (
                "labels.csv",
                b"filename,label\nimages/one.bmp,cats\nimages/two.tiff,dogs\n",
            ),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["classes"], ["cats", "dogs"])
        self.assertEqual(report["class_counts"], {"cats": 1, "dogs": 1})

    def test_ham10000_image_parts_use_image_id_and_dx_metadata(self):
        report = validate_upload(_zip_file([
            ("HAM10000_images_part_1/ISIC_0024306.jpg", _image_bytes()),
            ("HAM10000_images_part_1/ISIC_0024307.jpg", _image_bytes("black")),
            ("HAM10000_images_part_2/ISIC_0024308.jpg", _image_bytes("red")),
            (
                "HAM10000_metadata.csv",
                b"image_id,dx\nISIC_0024306,mel\nISIC_0024307,nv\nISIC_0024308,bcc\n",
            ),
            ("hmnist_28_28_L.csv", b"pixel_0,pixel_1\n0,1\n"),
        ]))

        self.assertTrue(report["valid"])
        self.assertEqual(report["classes"], ["bcc", "mel", "nv"])
        self.assertEqual(report["metadata"]["id_column"], "image_id")
        self.assertEqual(report["metadata"]["label_column"], "dx")
        self.assertIn("hmnist_28_28_L.csv", report["metadata"]["additional_files"])

    def test_non_zip_upload_is_rejected(self):
        report = validate_upload(SimpleUploadedFile("dataset.tar", b"archive"))

        self.assertFalse(report["valid"])
        self.assertIn("labelled image folders", report["errors"][0])

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


class DatasetUploadViewTests(TestCase):
    def test_upload_view_rejects_non_zip_before_creating_dataset(self):
        response = self.client.post(
            reverse("datasets:upload"),
            {"name": "Wrong format", "dataset_file": SimpleUploadedFile("dataset.tar", b"archive")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload a ZIP dataset")
        self.assertFalse(Dataset.objects.exists())

    def test_upload_view_rejects_files_over_the_configured_limit(self):
        from unittest.mock import patch

        with patch("apps.datasets.views.MAX_ARCHIVE_BYTES", 1):
            response = self.client.post(
                reverse("datasets:upload"),
                {"name": "Too large", "dataset_file": SimpleUploadedFile("dataset.zip", b"12")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "7 GB upload limit")
        self.assertFalse(Dataset.objects.exists())


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

class DatasetCleaningTests(SimpleTestCase):
    def test_clean_records_applies_every_removal_rule(self):
        records = [
            {"filename": "keep.png", "label": "benign"},
            {"filename": "dup.png", "label": "benign", "duplicate": True},
            {"filename": "near.png", "label": "benign", "near_duplicate": True},
            {"filename": "small.png", "label": "benign", "low_quality": True},
            {"filename": "unlabelled.png", "label": ""},
        ]

        kept, removed = clean_records(records)

        self.assertEqual([record["filename"] for record in kept], ["keep.png"])
        self.assertEqual(removed["duplicate"], ["dup.png"])
        self.assertEqual(removed["near_duplicate"], ["near.png"])
        self.assertEqual(removed["low_quality"], ["small.png"])
        self.assertEqual(removed["unlabelled"], ["unlabelled.png"])

    def test_cleaning_report_counts_removals_and_rechecks_the_manifest(self):
        image = _image_bytes()
        validation = validate_upload(_zip_file([
            ("benign/one.png", image),
            ("benign/duplicate.png", image),
            ("malignant/two.png", _image_bytes("black")),
            ("notes.txt", b"not an image or metadata"),
        ]))

        report = build_cleaning_report(validation)

        self.assertEqual(report["audited_images"], 3)
        self.assertEqual(report["unknown_file_count"], 1)
        self.assertEqual(report["kept_count"], 2)
        self.assertEqual(report["removed_count"], 1)
        self.assertEqual(report["classes_after"], ["benign", "malignant"])
        self.assertTrue(report["recheck"]["passed"])
        self.assertTrue(
            any(
                action["key"] == "duplicate" and action["removed"] == 1
                for action in report["actions"]
            )
        )

    def test_unlabelled_images_are_excluded_and_dataset_stays_ready(self):
        validation = validate_upload(_zip_file([
            ("benign/one.png", _image_bytes()),
            ("benign/two.png", _image_bytes("gray")),
            ("malignant/one.png", _image_bytes("black")),
            ("loose.png", _image_bytes("red")),
        ]))

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["unlabelled_count"], 1)
        report = build_cleaning_report(validation)
        self.assertEqual(report["kept_count"], 3)
        self.assertTrue(report["recheck"]["passed"])
        removed = {action["key"]: action["removed"] for action in report["actions"]}
        self.assertEqual(removed["unlabelled"], 1)

    def test_recheck_fails_when_cleaning_empties_a_class(self):
        validation = validate_upload(_zip_file([
            ("benign/one.png", _image_bytes()),
            ("malignant/small.png", _image_bytes("black", size=(16, 16))),
        ]))

        report = build_cleaning_report(validation)

        self.assertEqual(report["kept_count"], 1)
        self.assertFalse(report["recheck"]["passed"])
        self.assertEqual(report["dropped_classes"], ["malignant"])

    def test_recheck_flags_rare_classes_as_warning_only(self):
        # Solid grays spaced 60 luminance levels apart never collide as
        # near-duplicates; the striped pattern guarantees a distant hash.
        def _gray(level):
            return _image_bytes((level, level, level))

        striped = io.BytesIO()
        pattern = Image.new("L", (64, 48), 0)
        for x in range(0, 64, 8):
            for column in range(x, x + 4):
                for y in range(48):
                    pattern.putpixel((column, y), 255)
        pattern.save(striped, format="PNG")

        validation = validate_upload(_zip_file([
            ("benign/one.png", _gray(0)),
            ("benign/two.png", _gray(60)),
            ("benign/three.png", _gray(120)),
            ("benign/four.png", _gray(180)),
            ("benign/five.png", _gray(240)),
            ("malignant/one.png", striped.getvalue()),
        ]))

        report = build_cleaning_report(validation)

        self.assertEqual(report["kept_count"], 6)
        self.assertEqual(report["rare_classes"], {"malignant": 1})
        self.assertTrue(report["recheck"]["passed"])
        rare_check = next(
            check for check in report["recheck"]["checks"]
            if check["name"] == "Rare classes"
        )
        self.assertEqual(rare_check["severity"], "warning")
        self.assertFalse(rare_check["passed"])


class AuditIssueListTests(SimpleTestCase):
    def test_issue_list_reports_counts_severity_and_actions(self):
        image = _image_bytes()
        validation = validate_upload(_zip_file([
            ("benign/one.png", image),
            ("benign/duplicate.png", image),
            ("malignant/two.png", _image_bytes("black")),
            ("notes.txt", b"not an image or metadata"),
        ]))
        profile = build_profile(validation)
        leakage = analyze_leakage(validation)
        bias = analyze_bias(profile)
        imbalance = analyze_imbalance(validation["class_counts"])
        audit = build_audit(validation, profile, leakage, bias, imbalance)
        issues = {issue["title"]: issue for issue in audit["issues"]}

        self.assertEqual(issues["Exact duplicate images"]["count"], 1)
        self.assertEqual(issues["Exact duplicate images"]["severity"], "Medium")
        self.assertIn("Removed automatically", issues["Exact duplicate images"]["action"])
        self.assertEqual(issues["Unknown or unsupported files"]["count"], 1)
        self.assertEqual(issues["Missing class labels"]["severity"], "None")
        self.assertEqual(issues["Missing class labels"]["action"], "No action needed")
        self.assertGreaterEqual(audit["issue_count"], 1)
        self.assertIn(audit["highest_severity"], {"High", "Medium", "Low"})
        self.assertEqual(audit["issues"][0]["severity"], audit["highest_severity"])


class DatasetPipelineCleaningTests(TestCase):
    def test_pipeline_stores_cleaning_report_and_recheck(self):
        image = _image_bytes()
        dataset = Dataset.objects.create(
            name="Cleaning pipeline",
            uploaded_file=_zip_file([
                ("benign/one.png", image),
                ("benign/duplicate.png", image),
                ("malignant/two.png", _image_bytes("black")),
                ("notes.txt", b"not an image or metadata"),
            ]),
        )

        result = run_dataset_pipeline(dataset)

        self.assertEqual(result.status, "ready")
        cleaning = result.audit["cleaning"]
        self.assertEqual(cleaning["kept_count"], 2)
        self.assertEqual(cleaning["removed_count"], 1)
        self.assertEqual(cleaning["unknown_file_count"], 1)
        self.assertTrue(cleaning["recheck"]["passed"])
        self.assertEqual(result.pipeline["workflow"]["cleaning"]["status"], "completed")
        self.assertEqual(result.pipeline["workflow"]["recheck"]["status"], "passed")
        self.assertEqual(result.pipeline["configuration"]["cleaned_sample_count"], 2)

    def test_pipeline_requires_recheck_to_pass_before_ready(self):
        dataset = Dataset.objects.create(
            name="Emptied class",
            uploaded_file=_zip_file([
                ("benign/one.png", _image_bytes()),
                ("malignant/small.png", _image_bytes("black", size=(16, 16))),
            ]),
        )

        result = run_dataset_pipeline(dataset)

        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.pipeline["status"], "needs_review")
        self.assertFalse(result.audit["cleaning"]["recheck"]["passed"])
        self.assertEqual(result.pipeline["workflow"]["recheck"]["status"], "needs_review")

    def test_audit_view_exposes_cleaning_context_and_panel(self):
        image = _image_bytes()
        dataset = Dataset.objects.create(
            name="Cleaning view",
            uploaded_file=_zip_file([
                ("benign/one.png", image),
                ("benign/duplicate.png", image),
                ("malignant/two.png", _image_bytes("black")),
            ]),
        )
        run_dataset_pipeline(dataset)

        response = self.client.get(reverse("datasets:audit", args=[dataset.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context["cleaning"])
        self.assertTrue(response.context["progress"]["cleaned"])
        self.assertContains(response, "Automatic cleaning")
        self.assertContains(response, "Issues found")
        self.assertContains(response, "Cleaned, training-ready dataset")
