"""Build live workspace coverage data for the overview dataset radar."""

from django.utils import timezone


DOMAIN_CONFIG = {
    "health": {
        "name": "Skin & health imaging",
        "empty_description": "No health datasets detected yet. Upload labeled medical images to activate this domain.",
        "keywords": (
            "skin", "derma", "dermat", "lesion", "melanoma", "nevus", "tumor",
            "cancer", "medical", "health", "clinical", "x-ray", "xray", "disease",
        ),
    },
    "environment": {
        "name": "Plant & environment",
        "empty_description": "No environment datasets detected yet. Upload labeled plant or field images to activate this domain.",
        "keywords": (
            "plant", "crop", "leaf", "leaves", "flower", "agriculture", "soil",
            "field", "forest", "tree", "weed", "environment", "wildlife",
        ),
    },
    "vision": {
        "name": "General visual research",
        "empty_description": "No general vision datasets detected yet. Upload a labeled image set to activate this domain.",
        "keywords": (),
    },
}


def _dataset_classes(dataset):
    """Return the labels we can safely count, including older profile records."""
    classes = dataset.classes
    if isinstance(classes, dict):
        classes = classes.keys()
    if not isinstance(classes, (list, tuple, set)):
        classes = []
    if not classes and isinstance(dataset.profile, dict):
        distribution = dataset.profile.get("class_distribution", {})
        if isinstance(distribution, dict):
            classes = distribution.keys()
    return {str(label).strip() for label in classes if str(label).strip()}


def _classify_dataset(dataset):
    searchable = " ".join(
        [dataset.name, *_dataset_classes(dataset)]
    ).casefold()
    matched = {
        domain_key: {
            keyword for keyword in config["keywords"] if keyword in searchable
        }
        for domain_key, config in DOMAIN_CONFIG.items()
    }
    health_specific = matched["health"] - {"health", "disease"}
    if matched["environment"] and not health_specific:
        return "environment"
    if matched["health"]:
        return "health"
    if matched["environment"]:
        return "environment"
    return "vision"


def _domain_description(config, dataset_count, sample_count, class_count):
    if not dataset_count:
        return config["empty_description"]
    dataset_label = "dataset" if dataset_count == 1 else "datasets"
    image_label = "image" if sample_count == 1 else "images"
    class_label = "label class" if class_count == 1 else "label classes"
    return (
        f"Live coverage from {dataset_count} workspace {dataset_label}, "
        f"with {class_count} {class_label} across {sample_count} {image_label}."
    )


def build_radar_data(datasets):
    """Aggregate the current dataset table into JSON-safe radar data."""
    domain_state = {
        key: {
            "name": config["name"],
            "dataset_count": 0,
            "sample_count": 0,
            "class_names": set(),
        }
        for key, config in DOMAIN_CONFIG.items()
    }
    all_class_names = set()
    total_samples = 0
    dataset_list = list(datasets)

    for dataset in dataset_list:
        domain_key = _classify_dataset(dataset)
        state = domain_state[domain_key]
        labels = _dataset_classes(dataset)
        state["dataset_count"] += 1
        state["sample_count"] += dataset.sample_count or 0
        state["class_names"].update(labels)
        all_class_names.update(labels)
        total_samples += dataset.sample_count or 0

    domains = {}
    for key, state in domain_state.items():
        class_count = len(state["class_names"])
        domains[key] = {
            "name": state["name"],
            "dataset_count": state["dataset_count"],
            "sample_count": state["sample_count"],
            "class_count": class_count,
            "description": _domain_description(
                DOMAIN_CONFIG[key],
                state["dataset_count"],
                state["sample_count"],
                class_count,
            ),
        }

    primary_domain = max(
        domains,
        key=lambda key: (domains[key]["dataset_count"], domains[key]["sample_count"]),
    )
    primary = domains[primary_domain]
    total_dataset_count = len(dataset_list)
    if total_dataset_count:
        dataset_label = "dataset" if total_dataset_count == 1 else "datasets"
        footnote = (
            f"Radar is synced from {total_dataset_count} workspace {dataset_label} "
            f"and {total_samples} indexed images."
        )
    else:
        footnote = "Radar is ready for live coverage. Add a labeled dataset to activate workspace signals."

    return {
        "domains": domains,
        "primary_domain": primary_domain,
        "primary": primary,
        "total_dataset_count": total_dataset_count,
        "total_sample_count": total_samples,
        "total_class_count": len(all_class_names),
        "synced_label": "Synced just now",
        "synced_at": timezone.localtime().isoformat(),
        "footnote": footnote,
    }