"""Safe, archive-only validation for labelled image classification datasets."""
import hashlib
import io
import zipfile
from pathlib import PurePosixPath

from PIL import Image

from .metadata import attach_metadata, read_metadata


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILES = 10000
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MIN_IMAGE_DIMENSION = 32


def _label_for_member(name):
    path = PurePosixPath(name)
    if len(path.parts) < 2:
        return ""
    return path.parts[-2].strip()


def validate_upload(upload):
    """Inspect an uploaded ZIP without extracting it to the workspace."""
    report = {
        "valid": False,
        "format": "zip",
        "sample_count": 0,
        "readable_count": 0,
        "invalid_count": 0,
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
        report["errors"].append("The dataset exceeds the 2 GB upload limit.")
        return report
    if not upload.name.lower().endswith(".zip"):
        report["errors"].append("Upload a ZIP containing one folder per class.")
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
            seen_hashes = set()
            for info in members:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    report["errors"].append(f"Unsafe archive path: {info.filename}")
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                report["sample_count"] += 1
                label = _label_for_member(info.filename)
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
                        "low_quality": low_quality,
                    })
                    report["readable_count"] += 1
                    if label:
                        report["class_counts"][label] = report["class_counts"].get(label, 0) + 1
                    if duplicate:
                        report["duplicate_count"] += 1
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
        report["errors"].append("No supported JPG, PNG, or WEBP images were found.")
    if report["readable_count"] and len(report["classes"]) < 2:
        report["errors"].append("At least two labelled class folders are required.")
    if report["sample_count"] != report["readable_count"]:
        report["warnings"].append("Some images could not be validated.")
    report["valid"] = bool(
        report["readable_count"]
        and len(report["classes"]) >= 2
        and not report["errors"]
    )
    return report
