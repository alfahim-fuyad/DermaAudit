"""Deterministic split utilities for image-classification datasets.

These helpers mirror the logic in apps.training.services.trainer but are
exposed as a standalone module for reuse in notebooks and external scripts.
"""
import random
from typing import Dict, List, Tuple

RANDOM_SEED = 42
DEFAULT_SPLIT = (70, 15, 15)


def _normalize_split(split: Tuple[int, int, int]) -> Tuple[int, int, int]:
    values = tuple(int(v) for v in split)
    if len(values) != 3 or any(v < 1 for v in values) or sum(values) != 100:
        raise ValueError("Splits must be at least 1% and sum to 100%.")
    return values


def _split_counts(count: int, split: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Allocate a class across splits using largest-remainder rounding."""
    split = _normalize_split(split)
    if count <= 1:
        return count, 0, 0
    if count == 2:
        train_r, val_r, test_r = split
        return (1, 1, 0) if val_r >= test_r else (1, 0, 1)

    ratios = split
    quotas = [count * r / 100 for r in ratios]
    counts = [int(q) for q in quotas]
    for idx in sorted(range(3), key=lambda i: (quotas[i] - counts[i], -i), reverse=True):
        if sum(counts) >= count:
            break
        counts[idx] += 1

    for target in (1, 2):
        if counts[target]:
            continue
        donors = [i for i in range(3) if i != target and counts[i] > 1]
        if donors:
            donor = max(donors, key=counts.__getitem__)
            counts[donor] -= 1
            counts[target] += 1
    return tuple(counts)


def stratified_split(
    records: List[Dict], split: Tuple[int, int, int] = DEFAULT_SPLIT, seed: int = RANDOM_SEED
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create deterministic, class-aware train/val/test partitions."""
    split = _normalize_split(split)
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        grouped.setdefault(rec["label"], []).append(rec)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        c = len(items)
        train_c, val_c, test_c = _split_counts(c, split)
        train_c = c - test_c - val_c
        train.extend(items[:train_c])
        val.extend(items[train_c : train_c + val_c])
        test.extend(items[train_c + val_c :])
    return train, val, test


def group_aware_split(
    records: List[Dict], split: Tuple[int, int, int] = DEFAULT_SPLIT, seed: int = RANDOM_SEED
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Keep records from same entity together when group_id is available."""
    split = _normalize_split(split)
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        grouped.setdefault(rec.get("group_id") or f"__record__{rec['filename']}", []).append(rec)

    rng = random.Random(seed)
    groups = list(grouped.values())
    rng.shuffle(groups)
    total = sum(len(g) for g in groups)
    targets = [total * r / 100 for r in split]
    partitions: List[List[Dict]] = [[], [], []]
    counts = [0, 0, 0]
    for items in groups:
        idx = min(range(3), key=lambda i: (counts[i] - targets[i], i))
        partitions[idx].extend(items)
        counts[idx] += len(items)
    return tuple(partitions)
