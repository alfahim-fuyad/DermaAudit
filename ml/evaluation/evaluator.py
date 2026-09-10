"""Evaluation helpers for trained checkpoints.

This module wraps the training service's evaluation logic so that external
callers can evaluate a model without importing the full training pipeline.
"""
from typing import Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def evaluate_model(
    model,
    data_loader,
    labels: List[str],
    class_count: int,
) -> Dict:
    """Evaluate a model and return aggregate metrics.

    Mirrors apps.training.services.trainer._evaluate but exposed publicly.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for evaluation.")

    try:
        from apps.training.services.trainer import _evaluate

        return _evaluate(torch, model, data_loader, labels, class_count)
    except ImportError:
        # Minimal fallback
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        if not data_loader:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "auroc": None,
                "per_class": [],
                "confusion_matrix": [],
                "sample_count": 0,
            }

        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for images, targets in data_loader:
                all_logits.append(model(images))
                all_targets.append(targets)
        if not all_logits:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "auroc": None,
                "per_class": [],
                "confusion_matrix": [],
                "sample_count": 0,
            }
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        preds = logits.argmax(dim=1).tolist()
        actual = targets.tolist()

        return {
            "accuracy": float(accuracy_score(actual, preds)),
            "macro_f1": float(
                f1_score(actual, preds, average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(balanced_accuracy_score(actual, preds)),
            "precision": float(
                precision_score(actual, preds, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(actual, preds, average="macro", zero_division=0)
            ),
            "per_class": [],
            "confusion_matrix": confusion_matrix(actual, preds).tolist(),
            "sample_count": len(actual),
        }


def collect_outputs(model, data_loader):
    """Collect logits and targets from a data loader."""
    if torch is None:
        raise RuntimeError("PyTorch is required.")

    try:
        from apps.training.services.trainer import _collect_outputs

        return _collect_outputs(torch, model, data_loader)
    except ImportError:
        if not data_loader:
            return None, None
        model.eval()
        logits, actual = [], []
        with torch.no_grad():
            for images, targets in data_loader:
                logits.append(model(images))
                actual.append(targets)
        if not logits:
            return None, None
        return torch.cat(logits), torch.cat(actual)
