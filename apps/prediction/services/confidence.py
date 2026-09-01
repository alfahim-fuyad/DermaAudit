"""Decision policy for turning model probabilities into review states."""

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MARGIN_THRESHOLD = 0.10


def decide_confidence(
    confidence,
    ranked_predictions,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold=DEFAULT_MARGIN_THRESHOLD,
):
    """Return an explicit accept/abstain decision, never a silent guess."""
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    second_probability = 0.0
    if len(ranked_predictions or []) > 1:
        try:
            second_probability = float(ranked_predictions[1].get("probability", 0))
        except (TypeError, ValueError):
            second_probability = 0.0
    margin = max(0.0, confidence - second_probability)
    accepted = confidence >= threshold and margin >= margin_threshold
    return {
        "decision": "accept" if accepted else "abstain",
        "review_required": True,
        "threshold": threshold,
        "margin_threshold": margin_threshold,
        "margin": round(margin, 4),
        "reason": (
            "Confidence and top-class margin meet the configured thresholds."
            if accepted else
            "Confidence is below the configured threshold or the top classes are too close."
        ),
    }
