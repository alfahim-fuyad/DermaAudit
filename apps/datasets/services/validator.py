"""Safe, archive-only validation for labelled image classification datasets."""
import hashlib
import io
import zipfile
from pathlib import PurePosixPath

from PIL import Image

from .metadata import attach_metadata, read_metadata, _record_keys


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}
MAX_FILES = 10000
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_BYTES = 7 * 1024 * 1024 * 1024
MIN_IMAGE_DIMENSION = 32
GENERIC_LABEL_FOLDERS = {
    "data", "dataset", "files", "image", "images", "photo", "photos",
    "pictures", "test", "train", "training", "val", "valid", "validation",
}
IMAGE_CONTAINER_MARKERS = ("images_part", "image_part", "imageset", "image_set")


def _label_for_member(name):
    path = PurePosixPath(name)
    if len(path.parts) < 2:
        return ""
    return path.parts[-2].strip()


def _metadata_labels(summary, rows):
    """Index optional metadata labels for datasets with flat image archives."""
    if not summary.get("available") or not summary.get("id_column") or not summary.get("label_column"):
        return {}

    labels = {}
    for row in rows:
        label = str(row.get(summary["label_column"]) or "").strip()
        identifier = row.get(summary["id_column"])
        if not label or identifier in (None, ""):
            continue
        for key in _record_keys(str(identifier)):
            labels.setdefault(key, label)
    return labels


def _is_image_container(folder_label):
    normalized = folder_label.casefold().replace("-", "_")
    return normalized in GENERIC_LABEL_FOLDERS or any(
        marker in normalized for marker in IMAGE_CONTAINER_MARKERS
    )


def _difference_hash(image, size=16):
    """Return a small perceptual hash for near-duplicate detection."""
    image = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    return (
        "".join(
            "1" if pixels[row * (size + 1) + column] > pixels[row * (size + 1) + column + 1] else "0"
            for row in range(size)
            for column in range(size)
        ),
        sum(pixels) / len(pixels),
    )


def _hash_distance(first, second):
    return sum(left != right for left, right in zip(first, second))


def validate_upload(upload):
    """Inspect an uploaded ZIP without extracting it to the workspace."""
    report = {
        "valid": False,
        "format": "zip",
        "sample_count": 0,
        "readable_count": 0,
        "invalid_count": 0,
        "unlabelled_count": 0,
        "classes": [],
        "class_counts": {},
        "duplicate_count": 0,
        "near_duplicate_count": 0,
        "low_quality_count": 0,
        "errors": [],
        "warnings": [],
        "invalid_files": [],
        "records": [],
        "metadata": {},
        "_metadata_rows": [],
    }
    if not upload:
        report["errors"].append("No dataset file was provided.")
        return report
    if getattr(upload, "size", 0) > MAX_ARCHIVE_BYTES:
        report["errors"].append("The dataset exceeds the 7 GB upload limit.")
        return report
    if not upload.name.lower().endswith(".zip"):
        report["errors"].append(
            "Upload a ZIP containing labelled image folders or a metadata file with image labels."
        )
        return report

    try:
        upload.seek(0)
        with zipfile.ZipFile(upload) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir() and not info.filename.startswith("__MACOSX/")
            ]
            if len(members) > MAX_FILES:
                report["errors"].append(f"The archive contains more than {MAX_FILES:,} files.")
                return report

            metadata, metadata_rows = read_metadata(archive)
            report["metadata"] = metadata
            report["_metadata_rows"] = metadata_rows
            report["warnings"].extend(metadata.get("warnings", []))
            metadata_labels = _metadata_labels(metadata, metadata_rows)
            seen_hashes = set()
            perceptual_hashes = []
            for info in members:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    report["errors"].append(f"Unsafe archive path: {info.filename}")
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                report["sample_count"] += 1
                folder_label = _label_for_member(info.filename)
                metadata_label = next(
                    (metadata_labels[key] for key in _record_keys(info.filename) if key in metadata_labels),
                    "",
                )
                # A generic container such as images/ or train/ is not a
                # reliable class label when a manifest provides the label.
                label = (
                    metadata_label
                    if metadata_label and (
                        not folder_label
                        or _is_image_container(folder_label)
                    )
                    else folder_label
                )
                if not label:
                    report["warnings"].append(
                        f"Image is not inside a class folder: {info.filename}"
                    )
                if info.file_size > MAX_IMAGE_BYTES:
                    report["invalid_count"] += 1
                    report["invalid_files"].append(info.filename)
                    report["errors"].append(f"Image is too large: {info.filename}")
                    continue
                try:
                    with archive.open(info) as member:
                        data = member.read(MAX_IMAGE_BYTES + 1)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError("image exceeds the size limit")
                    digest = hashlib.sha256(data).hexdigest()
                    duplicate = digest in seen_hashes
                    seen_hashes.add(digest)
                    with Image.open(io.BytesIO(data)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(data)) as image:
                        width, height = image.size
                        mode = image.mode
                        metadata_count = len(image.info or {})
                        perceptual_hash = _difference_hash(image)
                    near_duplicate = any(
                        _hash_distance(perceptual_hash[0], previous[0]) <= 8
                        and abs(perceptual_hash[1] - previous[1]) <= 32
                        for previous in perceptual_hashes
                    )
                    perceptual_hashes.append(perceptual_hash)
                    low_quality = min(width, height) < MIN_IMAGE_DIMENSION
                    report["records"].append({
                        "filename": info.filename,
                        "label": label,
                        "extension": path.suffix.lower().lstrip("."),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "metadata_count": metadata_count,
                        "digest": digest,
                        "duplicate": duplicate,
                        "perceptual_hash": perceptual_hash,
                        "near_duplicate": near_duplicate and not duplicate,
                        "low_quality": low_quality,
                    })
                    report["readable_count"] += 1
                    if label:
                        report["class_counts"][label] = report["class_counts"].get(label, 0) + 1
                    else:
                        report["unlabelled_count"] += 1
                    if duplicate:
                        report["duplicate_count"] += 1
                    if near_duplicate and not duplicate:
                        report["near_duplicate_count"] += 1
                        if len(report["warnings"]) < 25:
                            report["warnings"].append(
                                f"Near-duplicate image content detected: {info.filename}"
                            )
                    if low_quality:
                        report["low_quality_count"] += 1
                        if len(report["warnings"]) < 25:
                            report["warnings"].append(
                                f"Image is smaller than {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION}: "
                                f"{info.filename}"
                            )
                except Exception as exc:
                    report["invalid_count"] += 1
                    if len(report["invalid_files"]) < 25:
                        report["invalid_files"].append(info.filename)
                    if len(report["errors"]) < 25:
                        report["errors"].append(f"Unreadable image {info.filename}: {exc}")
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        report["errors"].append(f"Could not read the ZIP archive: {exc}")
    finally:
        try:
            upload.seek(0)
        except (AttributeError, OSError):
            pass

    report["classes"] = sorted(report["class_counts"])
    attach_metadata(
        report["records"],
        report["metadata"],
        report.pop("_metadata_rows", []),
    )
    report["metadata"].pop("warnings", None)
    if not report["sample_count"]:
        report["errors"].append("No supported raster images were found.")
    if report["readable_count"] and len(report["classes"]) < 2:
        report["errors"].append("At least two labelled classes are required.")
    if report["sample_count"] != report["readable_count"]:
        report["warnings"].append("Some images could not be validated.")
    if report["unlabelled_count"]:
        report["errors"].append(
            f"{report['unlabelled_count']:,} readable image(s) do not have a class label."
        )
    report["valid"] = bool(
        report["readable_count"]
        and len(report["classes"]) >= 2
        and not report["errors"]
    )
    return report
