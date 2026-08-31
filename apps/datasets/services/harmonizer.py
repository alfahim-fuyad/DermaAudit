"""Normalize labels while preserving their original names for traceability."""
import re


def _normalized_label(label):
    return re.sub(r"\s+", " ", label.strip())


def harmonize_labels(classes):
    mapping = {_class: _normalized_label(_class) for _class in classes}
    return {
        "mapping": mapping,
        "classes": sorted(set(mapping.values())),
        "source_tag": "uploaded_archive",
    }
