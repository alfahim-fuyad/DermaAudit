"""Combine validation signals into a transparent, persisted audit report."""


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
            "stage_5": "completed" if readable else "blocked",
        },
    }
