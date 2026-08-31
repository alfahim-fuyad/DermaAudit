"""Command-line entry point for the same first-pass dataset checks used by the UI."""
import argparse
from pathlib import Path
from PIL import Image


def audit_folder(folder):
    folder = Path(folder)
    images = [path for path in folder.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    classes = sorted({path.parent.name for path in images})
    readable = 0
    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
            readable += 1
        except Exception:
            pass
    return {"samples": len(images), "readable": readable, "classes": classes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    print(audit_folder(args.folder))
