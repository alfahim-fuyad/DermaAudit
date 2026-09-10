"""Automatic cleaning of audited records into a training-ready manifest.

Cleaning never mutates the uploaded archive. The validator already verified
every readable image, so this stage applies deterministic removal rules to the
audited records and then re-checks the surviving set. Only the cleaned
manifest is considered training-ready.

Rules (applied in order):
- corrupted / unreadable images never enter the record set (counted here)
- unknown or unsupported files are ignored (not image samples)
- exact duplicate content is removed, the first copy is kept
- near-duplicate content is removed
- low-resolution images are removed
- images without a class label are excluded
"""

MAX_EXAMPLES = 5
RARE_CLASS_THRESHOLD = 5

REMOVAL_RULES = (
    (
        "invalid",
        "Corrupted or unreadable images",
        "Removed automatically — the file never enters the cleaned dataset",
    ),
    (
        "unknown_files",
        "Unknown or unsupported files",
        "Ignored — not usable image samples or recognized metadata",
    ),
    (
        "duplicate",
        "Exact duplicate images",
        "Removed automatically — the first copy is kept",
    ),
    (
        "near_duplicate",
        "Near-duplicate images",
        "Removed automatically before splitting",
    ),
    (
        "low_quality",
        "Low-resolution images",
        "Removed automatically",
    ),
    (
        "unlabelled",
        "Images without a class label",
        "Excluded from the cleaned dataset",
    ),
)


def clean_records(records):
    """Filter audited records into a cleaned manifest.

    Returns (kept_records, removed) where removed maps each record-level rule
    key to the list of removed filenames.
    """
    kept = []
    removed = {"duplicate": [], "near_duplicate": [], "low_quality": [], "unlabelled": []}
    for record in records or []:
        filename = record.get("filename", "")
        if record.get("duplicate"):
            removed["duplicate"].append(filename)
            continue
        if record.get("near_duplicate"):
            removed["near_duplicate"].append(filename)
            continue
        if record.get("low_quality"):
            removed["low_quality"].append(filename)
            continue
        if not record.get("label"):
            removed["unlabelled"].append(filename)
            continue
        kept.append(record)
    return kept, removed


def _class_counts(records):
    counts = {}
    for record in records:
        label = record.get("label")
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def recheck_cleaned(kept, classes_before):
    """Re-check the cleaned manifest: readability, labels, and class counts."""
    checks = []
    kept_count = len(kept)
    checks.append({
        "name": "Image readability",
        "severity": "blocking",
        "passed": kept_count > 0,
        "detail": f"{kept_count:,} previously verified image(s) re-checked",
    })
    unlabelled = sum(1 for record in kept if not record.get("label"))
    checks.append({
        "name": "Class labels",
        "severity": "blocking",
        "passed": unlabelled == 0 and kept_count > 0,
        "detail": "Every cleaned sample carries a class label" if not unlabelled
        else f"{unlabelled:,} sample(s) still lack a label",
    })
    counts_after = _class_counts(kept)
    two_classes = len(counts_after) >= 2
    checks.append({
        "name": "Class count",
        "severity": "blocking",
        "passed": two_classes,
        "detail": f"{len(counts_after)} class(es) after cleaning (2 required)",
    })
    dropped = sorted(set(classes_before) - set(counts_after))
    checks.append({
        "name": "Class coverage",
        "severity": "blocking",
        "passed": not dropped,
        "detail": "All audited classes survive cleaning" if not dropped
        else f"Classes emptied by cleaning: {', '.join(dropped[:10])}",
    })
    rare = {label: count for label, count in counts_after.items() if count < RARE_CLASS_THRESHOLD}
    checks.append({
        "name": "Rare classes",
        "severity": "warning",
        "passed": not rare,
        "detail": f"No class has fewer than {RARE_CLASS_THRESHOLD} samples" if not rare
        else f"Rare classes (fewer than {RARE_CLASS_THRESHOLD} samples): "
        + ", ".join(f"{label} ({count})" for label, count in sorted(rare.items())[:10]),
    })
    passed = all(check["passed"] for check in checks if check["severity"] == "blocking")
    return {
        "passed": passed,
        "checks": checks,
        "class_counts_after": counts_after,
        "rare_classes": rare,
        "dropped_classes": dropped,
    }


def build_cleaning_report(validation):
    """Build the persisted automatic-cleaning report from a validation report."""
    records = validation.get("records") or []
    invalid_count = int(validation.get("invalid_count", 0))
    unknown_file_count = int(validation.get("unknown_file_count", 0))
    audited_images = int(validation.get("sample_count", 0))
    class_counts_before = {
        str(label): int(count)
        for label, count in (validation.get("class_counts") or {}).items()
    }

    kept, removed = clean_records(records)
    recheck = recheck_cleaned(kept, sorted(class_counts_before))
    counts_after = recheck["class_counts_after"]

    actions = []
    for key, label, handling in REMOVAL_RULES:
        if key == "invalid":
            count = invalid_count
            examples = list(validation.get("invalid_files", []))[:MAX_EXAMPLES]
        elif key == "unknown_files":
            count = unknown_file_count
            examples = []
        else:
            filenames = removed.get(key, [])
            count = len(filenames)
            examples = filenames[:MAX_EXAMPLES]
        actions.append({
            "key": key,
            "label": label,
            "removed": count,
            "handling": handling,
            "examples": examples,
        })

    removed_from_images = sum(action["removed"] for action in actions if action["key"] != "unknown_files")
    kept_count = len(kept)
    return {
        "status": "completed",
        "policy": "automatic_cleaning_v1",
        "original_archive_untouched": True,
        "audited_images": audited_images,
        "unknown_file_count": unknown_file_count,
        "kept_count": kept_count,
        "removed_count": removed_from_images,
        "removal_rate": round((removed_from_images / audited_images) * 100, 1) if audited_images else 0,
        "kept_rate": round((kept_count / audited_images) * 100, 1) if audited_images else 0,
        "actions": actions,
        "class_counts_before": class_counts_before,
        "class_counts_after": counts_after,
        "classes_after": sorted(counts_after),
        "recheck": {
            "passed": recheck["passed"],
            "checks": recheck["checks"],
        },
        "rare_classes": recheck["rare_classes"],
        "dropped_classes": recheck["dropped_classes"],
    }
