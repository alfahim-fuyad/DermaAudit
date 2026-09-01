"""Read optional metadata files from a dataset ZIP archive.

Metadata is optional. When it is present, this module only keeps a small
summary and uses matching rows to enrich image records for the audit.
"""
import csv
import io
import json
import re


MAX_METADATA_BYTES = 10 * 1024 * 1024
METADATA_NAME_HINTS = ("metadata", "label", "annotation", "manifest")
ID_COLUMN_HINTS = (
    "image_id",
    "image",
    "filename",
    "file",
    "filepath",
    "path",
    "id",
)
GROUP_COLUMN_HINTS = (
    "group_id",
    "patient_id",
    "subject_id",
    "case_id",
    "study_id",
    "entity_id",
)
LABEL_COLUMN_HINTS = ("label", "class", "category", "diagnosis", "target")
SUBGROUP_COLUMN_HINTS = (
    "age",
    "sex",
    "gender",
    "ethnicity",
    "race",
    "skin_type",
    "skin_tone",
    "site",
    "source",
)


def _normalise_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(columns, hints):
    normalised = {
        _normalise_column_name(column): column
        for column in columns
    }
    for hint in hints:
        hint_key = _normalise_column_name(hint)
        if hint_key in normalised:
            return normalised[hint_key]
    for key, original in normalised.items():
        if any(_normalise_column_name(hint) in key for hint in hints):
            return original
    return None


def _metadata_member(info):
    name = info.filename.rsplit("/", 1)[-1].lower()
    suffix = name.rsplit(".", 1)[-1] if "." in name else ""
    return suffix in {"csv", "json"} and any(
        hint in name for hint in METADATA_NAME_HINTS
    )


def _rows_from_json(raw):
    value = json.loads(raw.decode("utf-8-sig"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "images", "data", "annotations"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
        if value and all(isinstance(item, dict) for item in value.values()):
            return list(value.values())
    return []


def _rows_from_member(archive, info):
    with archive.open(info) as member:
        raw = member.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("metadata file exceeds the 10 MB limit")
    if info.filename.lower().endswith(".json"):
        return _rows_from_json(raw)
    text = raw.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def read_metadata(archive):
    """Return a summary and private rows for recognised metadata files."""
    summary = {
        "available": False,
        "files": [],
        "columns": [],
        "row_count": 0,
        "id_column": None,
        "label_column": None,
        "group_column": None,
        "subgroup_columns": [],
        "matched_images": 0,
        "unmatched_rows": 0,
        "label_mismatch_count": 0,
        "group_count": 0,
        "warnings": [],
    }
    rows = []

    for info in archive.infolist():
        if info.is_dir() or not _metadata_member(info):
            continue
        summary["files"].append(info.filename)
        try:
            rows.extend(_rows_from_member(archive, info))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            summary["warnings"].append(
                f"Could not read metadata file {info.filename}: {exc}"
            )

    if not rows:
        summary["files"] = summary["files"][:5]
        return summary, rows

    columns = sorted({str(column) for row in rows for column in row})
    summary.update({
        "available": True,
        "columns": columns[:50],
        "row_count": len(rows),
        "id_column": _find_column(columns, ID_COLUMN_HINTS),
        "label_column": _find_column(columns, LABEL_COLUMN_HINTS),
        "group_column": _find_column(columns, GROUP_COLUMN_HINTS),
    })
    summary["subgroup_columns"] = [
        column for column in columns
        if any(
            _normalise_column_name(hint) in _normalise_column_name(column)
            for hint in SUBGROUP_COLUMN_HINTS
        )
    ][:20]
    return summary, rows


def _value_key(value):
    return str(value or "").strip().replace("\\", "/").lower()


def _record_keys(filename):
    path = _value_key(filename)
    basename = path.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    return {path, basename, stem}


def attach_metadata(records, summary, rows):
    """Match metadata rows to images and update the metadata summary."""
    if not summary.get("available") or not summary.get("id_column"):
        return

    rows_by_key = {}
    for row in rows:
        key = _value_key(row.get(summary["id_column"]))
        if key:
            rows_by_key.setdefault(key, row)

    matched_rows = set()
    group_ids = set()
    for record in records:
        row = next(
            (rows_by_key[key] for key in _record_keys(record["filename"]) if key in rows_by_key),
            None,
        )
        if row is None:
            continue
        matched_rows.add(id(row))
        if summary.get("label_column"):
            metadata_label = str(row.get(summary["label_column"]) or "").strip()
            record["metadata_label"] = metadata_label
            if metadata_label.casefold() != str(record.get("label") or "").casefold():
                summary["label_mismatch_count"] += 1
        if summary.get("group_column"):
            group_id = str(row.get(summary["group_column"]) or "").strip()
            if group_id:
                record["group_id"] = group_id
                group_ids.add(group_id)
        subgroup_values = {
            column: str(row.get(column) or "").strip()
            for column in summary.get("subgroup_columns", [])
            if str(row.get(column) or "").strip()
        }
        if subgroup_values:
            record["subgroups"] = subgroup_values

    summary["matched_images"] = len(matched_rows)
    summary["unmatched_rows"] = max(0, len(rows) - len(matched_rows))
    summary["group_count"] = len(group_ids)