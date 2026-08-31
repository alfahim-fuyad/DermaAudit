import io
import zipfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from PIL import Image

from .services.validator import validate_upload
from .services.profiler import build_profile
from .services.leakage import analyze_leakage
from .services.bias import analyze_bias
from .services.auditor import build_audit
from .models import Dataset
from .services.pipeline import run_dataset_pipeline


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