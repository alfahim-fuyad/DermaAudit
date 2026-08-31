"""Class distribution and weighted-sampling recommendations."""


def analyze_imbalance(class_counts):
    counts = {key: int(value) for key, value in class_counts.items()}
    if not counts:
        return {
            "class_counts": {},
            "minority_class": None,
            "majority_class": None,
            "ratio": None,
            "severity": "Unknown",
            "recommendation": "Add labelled class folders before training.",
        }
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    minimum = counts[minority_class]
    maximum = counts[majority_class]
    ratio_value = round(maximum / minimum, 1) if minimum else None
    severity = "Low" if ratio_value <= 1.5 else "Review" if ratio_value <= 3 else "High"
    return {
        "class_counts": counts,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "ratio": f"1:{ratio_value}" if ratio_value is not None else None,
        "severity": severity,
        "recommendation": (
            "Use weighted loss or balanced sampling during training."
            if severity != "Low" else
            "Class distribution is reasonably balanced."
        ),
    }
