"""Leakage checks that can be performed from an image archive alone."""


def analyze_leakage(validation):
    metadata = validation.get("metadata", {})
    duplicate_files = [
        record["filename"] for record in validation.get("records", [])
        if record.get("duplicate")
    ][:25]
    if duplicate_files:
        risk = "Review"
        recommendation = "Remove or group duplicate images before splitting."
    else:
        risk = "Low"
        recommendation = "No exact duplicate image content was detected."
    if metadata.get("group_column") and metadata.get("matched_images"):
        split_strategy = "group-aware"
        group_note = (
            f"Use {metadata['group_column']} for train, validation, and test splits "
            f"({metadata.get('group_count', 0)} groups detected)."
        )
        recommendation = f"{recommendation} {group_note}"
    else:
        split_strategy = "stratified"
        group_note = "No entity or group identifier was matched to the uploaded images."
    return {
        "group_information_available": bool(metadata.get("group_column") and metadata.get("matched_images")),
        "group_id_source": metadata.get("group_column"),
        "group_count": metadata.get("group_count", 0),
        "split_strategy": split_strategy,
        "risk": risk,
        "duplicate_count": validation.get("duplicate_count", 0),
        "duplicate_files": duplicate_files,
        "recommendation": recommendation,
        "note": group_note,
    }
