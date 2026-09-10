"""Small, explicit experiment protocol metadata for comparable model runs."""


EXPERIMENT_VARIANTS = (
    {
        "key": "original",
        "label": "Original",
        "intervention": "None",
        "description": "Use the uploaded samples with the shared training protocol.",
    },
    {
        "key": "duplicate_free",
        "label": "Duplicate-free",
        "intervention": "Remove exact duplicates",
        "description": "Exclude repeated image content before splitting.",
    },
    {
        "key": "leakage_controlled",
        "label": "Leakage-controlled",
        "intervention": "Group-aware split",
        "description": "Keep images from the same entity in one partition.",
    },
    {
        "key": "bias_mitigated",
        "label": "Bias-mitigated",
        "intervention": "Subgroup review",
        "description": "Track subgroup metrics when approved metadata is available.",
    },
    {
        "key": "quality_controlled",
        "label": "Quality-controlled",
        "intervention": "Quality review",
        "description": "Flag low-resolution and unreadable samples for review.",
    },
    {
        "key": "imbalance_handled",
        "label": "Imbalance-handled",
        "intervention": "Weighted loss",
        "description": "Use class weights to reduce majority-class dominance.",
    },
    {
        "key": "fully_audited",
        "label": "Fully audited",
        "intervention": "All available safeguards",
        "description": "Use the dataset configuration produced by the complete audit.",
    },
)


def prepare_records(records, variant):
    """Apply only the declared intervention for a controlled experiment."""
    records = list(records or [])
    if variant == "duplicate_free":
        return [
            record for record in records
            if not record.get("duplicate")
        ]
    if variant == "quality_controlled":
        return [
            record for record in records
            if not record.get("low_quality")
        ]
    if variant == "fully_audited":
        return [
            record for record in records
            if not record.get("duplicate")
            and not record.get("near_duplicate")
            and not record.get("low_quality")
        ]
    return records


def build_experiment_plan(completed_variant="original"):
    """Return JSON-safe experiment states without claiming unrun work completed."""
    return [
        {
            **variant,
            "status": "completed" if variant["key"] == completed_variant else "not_run",
        }
        for variant in EXPERIMENT_VARIANTS
    ]
