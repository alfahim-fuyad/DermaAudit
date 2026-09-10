"""Statistical helpers for model evaluation.

Provides deterministic, dependency-free utilities that complement
apps.reports.services.statistics.
"""
import random
from typing import Dict, List


def bootstrap_mean_interval(
    values: List[float], iterations: int = 400, seed: int = 42
) -> Dict:
    """Percentile bootstrap confidence interval for a mean."""
    numbers = [float(v) for v in values if v is not None]
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


def paired_permutation_test(
    a: List[float], b: List[float], iterations: int = 1000, seed: int = 42
) -> Dict:
    """Simple paired permutation test for difference in means."""
    if len(a) != len(b) or len(a) < 2:
        return {
            "status": "insufficient_data",
            "p_value": None,
            "detail": "Need paired samples of equal length >=2.",
        }

    rng = random.Random(seed)
    observed_diff = sum(x - y for x, y in zip(a, b)) / len(a)
    count_extreme = 0
    for _ in range(iterations):
        permuted = [
            (x - y) if rng.random() < 0.5 else (y - x) for x, y in zip(a, b)
        ]
        permuted_diff = sum(permuted) / len(permuted)
        if abs(permuted_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (iterations + 1)
    return {
        "status": "completed",
        "observed_diff": round(observed_diff, 6),
        "p_value": round(p_value, 6),
        "iterations": iterations,
        "method": "paired permutation",
    }
