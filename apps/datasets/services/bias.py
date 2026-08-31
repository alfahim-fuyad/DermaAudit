"""Metadata-aware bias analysis with an explicit unavailable state."""


def analyze_bias(profile):
    subgroup_columns = profile.get("subgroup_columns", [])
    if not subgroup_columns:
        return {
            "available": False,
            "status": "Not available",
            "subgroups": [],
            "recommendation": "Add approved subgroup metadata to enable fairness analysis.",
        }
    return {
        "available": True,
        "status": "Metadata detected; subgroup review required",
        "subgroups": subgroup_columns,
        "recommendation": "Review metadata fields and calculate subgroup metrics before release.",
    }
