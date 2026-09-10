"""Combine validation signals into a transparent, persisted audit report."""
from .cleaner import RARE_CLASS_THRESHOLD


SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "None": 3}


def _issue(title, count, severity, detail, action):
    """Return one audit-report issue row; clean categories keep a pass row."""
    active = bool(count) and count != "None"
    return {
        "title": title,
        "count": count,
        "severity": severity if active else "None",
        "detail": detail,
        "action": action if active else "No action needed",
    }


def build_issue_list(validation, profile, leakage, imbalance):
    """Structured issues grouped as image, label, and distribution quality."""
    class_counts = imbalance.get("class_counts", {}) or {}
    rare_classes = sum(1 for count in class_counts.values() if count < RARE_CLASS_THRESHOLD)
    imbalance_ratio = imbalance.get("ratio")
    imbalance_severity = {
        "High": "High",
        "Review": "Medium",
        "Low": "Low",
        "Unknown": "Medium",
    }.get(imbalance.get("severity"), "Low")
    leakage_risk = leakage.get("risk", "Low")
    leakage_severity = "High" if leakage_risk not in {"Low"} else "Low"

    issues = [
        # --- File / image quality ---
        _issue(
            "Corrupted or unreadable images",
            validation.get("invalid_count", 0),
            "High",
            "Files that failed format or image verification",
            "Excluded automatically — they never enter the cleaned dataset",
        ),
        _issue(
            "Unknown or unsupported files",
            validation.get("unknown_file_count", 0),
            "Low",
            "Archive members that are neither images nor recognized metadata",
            "Ignored automatically — not counted as image samples",
        ),
        _issue(
            "Exact duplicate images",
            validation.get("duplicate_count", 0),
            "Medium",
            "Identical file content detected by SHA-256 digest",
            "Removed automatically — the first copy is kept",
        ),
        _issue(
            "Near-duplicate images",
            validation.get("near_duplicate_count", 0),
            "Medium",
            "Perceptual hash similarity between different files",
            "Removed automatically before splitting",
        ),
        _issue(
            "Low-resolution images",
            validation.get("low_quality_count", 0),
            "Low",
            "Images smaller than the minimum accepted dimension",
            "Removed automatically from the cleaned dataset",
        ),
        # --- Label / data quality ---
        _issue(
            "Missing class labels",
            validation.get("unlabelled_count", 0),
            "High",
            "Readable images without a class folder or metadata label",
            "Excluded automatically from the cleaned dataset",
        ),
        _issue(
            "Metadata label mismatches",
            profile.get("metadata_label_mismatches", 0),
            "High",
            "Class folder and metadata label disagree for matched images",
            "Review the metadata file before training",
        ),
        # --- Distribution quality ---
        _issue(
            "Class imbalance",
            imbalance_ratio or 0,
            imbalance_severity,
            f"Largest class {imbalance.get('majority_class') or '—'} vs smallest "
            f"{imbalance.get('minority_class') or '—'}",
            "Handled by class-weighted loss and balanced sampling during training",
        ),
        _issue(
            "Rare classes",
            rare_classes,
            "Medium",
            f"Classes with fewer than {RARE_CLASS_THRESHOLD} samples",
            "Collect more samples or merge classes before relying on per-class metrics",
        ),
        _issue(
            "Leakage risk",
            1 if leakage_risk not in {"Low"} else 0,
            leakage_severity,
            f"Duplicate-content risk {leakage_risk} · {leakage.get('split_strategy')} split available",
            "Duplicates removed automatically; entities kept together by a group-aware split",
        ),
    ]
    return sorted(issues, key=lambda issue: SEVERITY_ORDER.get(issue["severity"], 9))


def build_audit(validation, profile, leakage, bias, imbalance):
    total = validation.get("sample_count", 0)
    readable = validation.get("readable_count", 0)
    invalid = validation.get("invalid_count", 0)
    duplicate_rate = round((validation.get("duplicate_count", 0) / total) * 100, 1) if total else 0
    near_duplicate_rate = round(
        (validation.get("near_duplicate_count", 0) / total) * 100, 1
    ) if total else 0
    quality_score = round((readable / total) * 100) if total else 0
    if invalid:
        quality_score = max(0, quality_score - min(20, invalid * 2))
    low_quality = validation.get("low_quality_count", 0)
    quality_score = max(0, quality_score - min(15, low_quality))
    label_mismatches = profile.get("metadata_label_mismatches", 0)
    unlabelled_images = validation.get("unlabelled_count", 0)
    labelled_images = max(0, readable - label_mismatches - unlabelled_images)
    label_consistency = round((labelled_images / readable) * 100) if readable else 0
    issues = build_issue_list(validation, profile, leakage, imbalance)
    active_issues = [issue for issue in issues if issue["severity"] != "None"]
    recommendation_parts = [leakage["recommendation"], imbalance["recommendation"]]
    if bias["available"]:
        recommendation_parts.append("Complete subgroup fairness analysis.")
    if low_quality:
        recommendation_parts.append("Review low-resolution images before training.")
    if validation.get("near_duplicate_count"):
        recommendation_parts.append("Review near-duplicate images before splitting.")
    if label_mismatches:
        recommendation_parts.append("Resolve image-label mismatches in the metadata.")
    return {
        "duplicate_rate": duplicate_rate,
        "near_duplicate_rate": near_duplicate_rate,
        "quality_score": quality_score,
        "leakage_risk": leakage["risk"],
        "label_consistency": label_consistency,
        "imbalance_ratio": imbalance["ratio"] or "Unknown",
        "issue_count": len(active_issues),
        "highest_severity": active_issues[0]["severity"] if active_issues else "None",
        "issues": issues,
        "recommendation": " ".join(recommendation_parts),
        "image_quality": {
            "readable": readable,
            "invalid": invalid,
            "near_duplicate": validation.get("near_duplicate_count", 0),
            "low_quality": low_quality,
            "unlabelled": unlabelled_images,
            "minimum_dimension": 32,
        },
        "leakage": leakage,
        "bias": bias,
        "imbalance": imbalance,
        "stages": {
            "validation": "passed" if validation["valid"] else "needs_review",
            "profiling": "completed" if readable else "blocked",
            "configuration": "completed" if readable and validation.get("classes") else "blocked",
            "harmonization": "completed" if validation.get("classes") else "blocked",
            "audit": "completed" if readable else "blocked",
            "leakage": "completed" if readable else "blocked",
            "bias": "completed" if readable else "blocked",
            "imbalance": "completed" if readable else "blocked",
            "cleaning": "completed" if readable else "blocked",
            "recheck": "completed" if readable else "blocked",
            "stage_5": "completed" if readable else "blocked",
        },
    }
