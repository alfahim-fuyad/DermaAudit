from PIL import Image, ImageStat


def analyze_image(uploaded_file):
    """Run a safe preview analysis until a registered checkpoint is available.

    A real deployment should replace this with a loaded Torchvision checkpoint.
    The returned flag keeps the UI honest: this is not a medical diagnosis.
    """
    image = Image.open(uploaded_file).convert("RGB")
    stat = ImageStat.Stat(image)
    brightness = sum(stat.mean) / 3
    red_bias = stat.mean[0] - ((stat.mean[1] + stat.mean[2]) / 2)
    if red_bias > 16 and brightness < 175:
        label, confidence = "Needs expert review", 0.58
    elif brightness > 220:
        label, confidence = "Low-signal image", 0.42
    else:
        label, confidence = "Needs expert review", 0.51
    return {
        "label": label,
        "confidence": confidence,
        "review_required": True,
        "mode": "preview_analysis",
        "note": "No trained checkpoint is registered. This preview is not a medical diagnosis.",
        "signals": ["Image format validated", "RGB channels detected", "Human review recommended"],
    }
