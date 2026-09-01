"""Aggregate image and label facts from validator records."""


def build_profile(validation):
    records = validation.get("records", [])
    metadata = validation.get("metadata", {})
    formats = sorted({record["extension"].upper() for record in records})
    channels = sorted({record["mode"] for record in records})
    widths = [record["width"] for record in records]
    heights = [record["height"] for record in records]
    metadata_images = sum(1 for record in records if record["metadata_count"])
    low_quality_images = sum(1 for record in records if record.get("low_quality"))
    if widths and heights:
        resolution = {
            "min": f"{min(widths)}×{min(heights)}",
            "max": f"{max(widths)}×{max(heights)}",
            "average": f"{round(sum(widths) / len(widths))}×{round(sum(heights) / len(heights))}",
        }
        resolution_label = resolution["average"]
    else:
        resolution = {}
        resolution_label = "No readable images"
    return {
        "formats": ", ".join(formats) or "No images found",
        "channels": ", ".join(channels) or "Not detected",
        "resolution": resolution_label,
        "resolution_stats": resolution,
        "metadata": "Detected" if metadata_images else "Not detected",
        "metadata_image_count": metadata_images,
        "class_distribution": validation.get("class_counts", {}),
        "readable_images": validation.get("readable_count", 0),
        "invalid_images": validation.get("invalid_count", 0),
        "unlabelled_images": validation.get("unlabelled_count", 0),
        "low_quality_images": low_quality_images,
        "metadata_files": metadata.get("files", []),
        "metadata_columns": metadata.get("columns", []),
        "metadata_rows": metadata.get("row_count", 0),
        "metadata_ids": metadata.get("id_column"),
        "group_information": bool(metadata.get("group_column")),
        "group_column": metadata.get("group_column"),
        "group_count": metadata.get("group_count", 0),
        "subgroup_columns": metadata.get("subgroup_columns", []),
        "metadata_label_mismatches": metadata.get("label_mismatch_count", 0),
    }
