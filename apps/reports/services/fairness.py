"""Metadata-aware subgroup metrics for evaluation records."""
from collections import defaultdict


def build_fairness_report(records, actual, predicted, labels):
    """Calculate simple per-subgroup accuracy when approved metadata exists."""
    groups = defaultdict(list)
    for record, expected, output in zip(records or [], actual or [], predicted or []):
        for column, value in (record.get("subgroups") or {}).items():
            groups[(column, value)].append(int(expected) == int(output))

    if not groups:
        return {
            "status": "not_available",
            "subgroups": [],
            "note": "Approved subgroup metadata was not available for evaluated images.",
        }

    subgroup_metrics = []
    for (column, value), outcomes in sorted(groups.items()):
        subgroup_metrics.append({
            "column": column,
            "value": value,
            "sample_count": len(outcomes),
            "accuracy": round(sum(outcomes) / len(outcomes), 4),
        })
    accuracies = [item["accuracy"] for item in subgroup_metrics]
    return {
        "status": "completed",
        "subgroups": subgroup_metrics,
        "accuracy_range": round(max(accuracies) - min(accuracies), 4),
        "labels": labels,
        "note": "Review subgroup sample sizes before interpreting differences.",
    }
