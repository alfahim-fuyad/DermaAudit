"""Synchronous dataset pipeline used by upload and re-audit actions."""
from django.db import transaction

from .auditor import build_audit
from .bias import analyze_bias
from .harmonizer import harmonize_labels
from .imbalance import analyze_imbalance
from .leakage import analyze_leakage
from .profiler import build_profile
from .validator import validate_upload


def _public_validation(report):
    return {
        key: value for key, value in report.items()
        if key != "records" and not key.startswith("_")
    }


@transaction.atomic
def run_dataset_pipeline(dataset):
    dataset.status = "auditing"
    dataset.pipeline = {
        "status": "running",
        "stages": ["validation", "profiling", "harmonization", "audit", "leakage", "bias", "imbalance"],
    }
    dataset.save(update_fields=["status", "pipeline", "updated_at"])
    try:
        with dataset.uploaded_file.open("rb") as upload:
            validation = validate_upload(upload)
        profile = build_profile(validation)
        harmonization = harmonize_labels(validation.get("classes", []))
        leakage = analyze_leakage(validation)
        bias = analyze_bias(profile)
        imbalance = analyze_imbalance(validation.get("class_counts", {}))
        audit = build_audit(validation, profile, leakage, bias, imbalance)
        dataset.sample_count = validation["sample_count"]
        dataset.class_count = len(harmonization["classes"])
        dataset.classes = harmonization["classes"]
        dataset.profile = {
            **profile,
            "validation": _public_validation(validation),
            "harmonization": harmonization,
        }
        dataset.validation = _public_validation(validation)
        dataset.audit = audit
        dataset.status = "ready" if validation["valid"] else "needs_review"
        dataset.pipeline = {
            "status": "completed" if validation["valid"] else "needs_review",
            "stages": audit["stages"],
        }
    except Exception as exc:
        dataset.status = "needs_review"
        dataset.validation = {
            "valid": False,
            "errors": [f"Pipeline failed: {exc}"],
            "warnings": [],
        }
        dataset.pipeline = {"status": "failed", "error": str(exc)}
    dataset.save()
    return dataset