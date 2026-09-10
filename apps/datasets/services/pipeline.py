"""Synchronous dataset pipeline used by upload and re-audit actions."""
from django.db import transaction

from .auditor import build_audit
from .bias import analyze_bias
from .cleaner import build_cleaning_report
from .harmonizer import harmonize_labels
from .imbalance import analyze_imbalance
from .leakage import analyze_leakage
from .profiler import build_profile
from .validator import validate_upload


DATASET_WORKFLOW_STAGES = (
    ("stage_0", "Stage 0 · Validation", "Format, image, label, class, metadata, and structure checks"),
    ("stage_1", "Stage 1 · Auto profiling", "Image, label, metadata, ID, group, and source profile"),
    ("configuration", "Automatic configuration", "Canonical labels, split policy, and imbalance strategy"),
    ("stage_2", "Stage 2 · Harmonization", "Standardized labels, metadata mapping, and source tags"),
    ("stage_3", "Stage 3 · Data audit", "Duplicates, quality, labels, metadata, and class distribution"),
    ("stage_4", "Stage 4 · Leakage control", "Group-aware or stratified split recommendation"),
    ("stage_5", "Stage 5 · Bias & imbalance", "Subgroup availability and imbalance handling"),
    ("cleaning", "Automatic cleaning", "Remove corrupted, duplicate, low-quality, and unlabelled samples"),
    ("recheck", "Cleaned dataset validation", "Re-check images, labels, and class counts after cleaning"),
)
WORKFLOW_VERSION = 1


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
            "workflow_version": WORKFLOW_VERSION,
        "stages": {key: "queued" for key, _label, _description in DATASET_WORKFLOW_STAGES},
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
        cleaning = build_cleaning_report(validation)
        audit["cleaning"] = cleaning
        recheck_passed = bool(cleaning.get("recheck", {}).get("passed"))
        configuration = {
            "task": "image_classification",
            "label_column": validation.get("metadata", {}).get("label_column"),
            "id_column": validation.get("metadata", {}).get("id_column"),
            "group_column": validation.get("metadata", {}).get("group_column"),
            "split_policy": leakage["split_strategy"],
            "split_ratios": [70, 15, 15],
            "label_mapping": harmonization["mapping"],
            "source_tag": harmonization["source_tag"],
            "imbalance_strategy": (
                "weighted_loss_and_balanced_sampling"
                if imbalance["severity"] != "Low" else "standard_sampling"
            ),
            "cleaning_policy": "automatic_cleaning_v1",
            "cleaned_sample_count": cleaning["kept_count"],
        }
        dataset.sample_count = validation["sample_count"]
        dataset.class_count = len(harmonization["classes"])
        dataset.classes = harmonization["classes"]
        dataset.profile = {
            **profile,
            "validation": _public_validation(validation),
            "harmonization": harmonization,
            "configuration": configuration,
        }
        dataset.validation = _public_validation(validation)
        dataset.audit = audit
        dataset.status = (
            "ready" if validation["valid"] and recheck_passed else "needs_review"
        )
        dataset.pipeline = {
            "status": (
                "completed" if validation["valid"] and recheck_passed else "needs_review"
            ),
            "workflow_version": WORKFLOW_VERSION,
            "stages": audit["stages"],
            "workflow": {
                "stage_0": {
                    "status": audit["stages"]["validation"],
                    "summary": f"{validation['readable_count']:,} readable image(s), "
                    f"{validation['invalid_count']:,} invalid",
                },
                "stage_1": {
                    "status": audit["stages"]["profiling"],
                    "summary": f"{profile['readable_images']:,} images profiled across "
                    f"{len(validation.get('classes', []))} classes",
                },
                "configuration": {
                    "status": audit["stages"]["configuration"],
                    "summary": f"{configuration['split_policy']} split · "
                    f"{configuration['imbalance_strategy'].replace('_', ' ')}",
                },
                "stage_2": {
                    "status": audit["stages"]["harmonization"],
                    "summary": f"{len(harmonization['classes'])} canonical classes",
                },
                "stage_3": {
                    "status": audit["stages"]["audit"],
                    "summary": f"Quality {audit['quality_score']}/100 · "
                    f"{validation['duplicate_count']} duplicate(s)",
                },
                "stage_4": {
                    "status": audit["stages"]["leakage"],
                    "summary": f"{leakage['split_strategy']} split · risk {leakage['risk']}",
                },
                "stage_5": {
                    "status": audit["stages"]["bias"],
                    "summary": f"Bias {bias['status']} · imbalance {imbalance['severity']}",
                },
                "cleaning": {
                    "status": audit["stages"]["cleaning"],
                    "summary": (
                        f"Removed {cleaning['removed_count']:,} of {cleaning['audited_images']:,} "
                        f"image(s) · {cleaning['kept_count']:,} kept"
                    ),
                },
                "recheck": {
                    "status": "passed" if recheck_passed else "needs_review",
                    "summary": (
                        f"{cleaning['kept_count']:,} cleaned sample(s) re-checked · "
                        f"{len(cleaning['classes_after'])} class(es) survive"
                    ),
                },
            },
            "configuration": configuration,
        }
    except Exception as exc:
        dataset.status = "needs_review"
        dataset.validation = {
            "valid": False,
            "errors": [f"Pipeline failed: {exc}"],
            "warnings": [],
        }
        dataset.pipeline = {
            "status": "failed",
            "workflow_version": WORKFLOW_VERSION,
            "error": str(exc),
            "stages": {key: "failed" for key, _label, _description in DATASET_WORKFLOW_STAGES},
        }
    dataset.save()
    return dataset