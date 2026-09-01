"""Deterministic, dependency-free confidence intervals for run metrics."""
import random


def bootstrap_mean_interval(values, iterations=400, seed=42):
    """Return a percentile bootstrap interval, or an unavailable result."""
    numbers = [float(value) for value in values if value is not None]
    if len(numbers) < 2:
        return {"status": "insufficient_data", "lower": None, "upper": None}

    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        resample = [numbers[rng.randrange(len(numbers))] for _ in numbers]
        samples.append(sum(resample) / len(resample))
    samples.sort()
    return {
        "status": "completed",
        "lower": round(samples[int(len(samples) * 0.025)], 4),
        "upper": round(samples[int(len(samples) * 0.975)], 4),
        "sample_count": len(numbers),
        "method": "percentile bootstrap",
    }


def build_run_statistics(evaluation):
    """Describe what statistical evidence is available for one evaluation."""
    sample_count = int((evaluation or {}).get("sample_count") or 0)
    if sample_count < 2:
        return {
            "status": "insufficient_data",
            "sample_count": sample_count,
            "confidence_intervals": {},
        }
    return {
        "status": "descriptive",
        "sample_count": sample_count,
        "confidence_intervals": {
            "accuracy": "Available after repeated resampling of per-image outcomes.",
            "macro_f1": "Available after repeated resampling of per-image outcomes.",
        },
        "note": "This run stores aggregate metrics; no statistical test is claimed without paired outcomes.",
    }
