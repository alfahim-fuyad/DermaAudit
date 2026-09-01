"""Helpers for keeping calibration evidence explicit in reports."""


def build_calibration_report(calibration):
    calibration = calibration or {}
    return {
        "status": calibration.get("status", "not_available"),
        "ece": calibration.get("ece"),
        "brier": calibration.get("brier"),
        "bins": calibration.get("bins", []),
        "interpretation": (
            "Lower ECE and Brier scores indicate better calibrated confidence."
            if calibration.get("status") == "completed"
            else "Calibration requires a validation or test partition."
        ),
    }
