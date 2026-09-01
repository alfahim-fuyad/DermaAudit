"""CPU training runner for reproducible image-classification experiments."""
import io
import os
import random
import zipfile
from pathlib import Path

from django.conf import settings

from apps.datasets.services.validator import validate_upload
from apps.reports.services.calibration import build_calibration_report
from apps.reports.services.fairness import build_fairness_report
from apps.reports.services.statistics import build_run_statistics
from .experiments import build_experiment_plan


IMAGE_SIZE = 64
RANDOM_SEED = 42
EPOCHS = 5
MAX_BATCH_SIZE = 32
DEFAULT_SPLIT = (70, 15, 15)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _report_progress(progress_callback, step, percent, label, detail):
    if progress_callback:
        progress_callback({
            "step": step,
            "percent": max(0, min(100, int(percent))),
            "label": label,
            "detail": detail,
        })


def _normalize_split(split):
    try:
        values = tuple(int(value) for value in split)
    except (TypeError, ValueError) as exc:
        raise ValueError("Split values must be whole-number percentages.") from exc
    if len(values) != 3 or any(value < 1 for value in values) or sum(values) != 100:
        raise ValueError("Training, validation, and test splits must be at least 1% and total 100%.")
    return values


def _split_counts(count, split):
    """Allocate a class across splits using largest-remainder rounding."""
    train_ratio, validation_ratio, test_ratio = _normalize_split(split)
    if count <= 1:
        return count, 0, 0
    if count == 2:
        return (1, 1, 0) if validation_ratio >= test_ratio else (1, 0, 1)

    ratios = (train_ratio, validation_ratio, test_ratio)
    quotas = [count * ratio / 100 for ratio in ratios]
    counts = [int(quota) for quota in quotas]
    for index in sorted(
        range(3),
        key=lambda item: (quotas[item] - counts[item], -item),
        reverse=True,
    ):
        if sum(counts) >= count:
            break
        counts[index] += 1

    for target in (1, 2):
        if counts[target]:
            continue
        donors = [index for index in range(3) if index != target and counts[index] > 1]
        if donors:
            donor = max(donors, key=counts.__getitem__)
            counts[donor] -= 1
            counts[target] += 1
    return tuple(counts)


def _split_records(records, split=DEFAULT_SPLIT):
    """Create deterministic, class-aware train/validation/test partitions."""
    split = _normalize_split(split)
    grouped = {}
    for record in records:
        grouped.setdefault(record["label"], []).append(record)

    rng = random.Random(RANDOM_SEED)
    train, validation, test = [], [], []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        count = len(items)
        train_count, validation_count, test_count = _split_counts(count, split)
        train_count = count - test_count - validation_count
        train.extend(items[:train_count])
        validation.extend(items[train_count:train_count + validation_count])
        test.extend(items[train_count + validation_count:])

    return train, validation, test


def _group_split_records(records, split=DEFAULT_SPLIT):
    """Keep records from the same entity together when group IDs are available."""
    split = _normalize_split(split)
    grouped = {}
    for record in records:
        grouped.setdefault(record.get("group_id") or f"__record__{record['filename']}", []).append(record)

    rng = random.Random(RANDOM_SEED)
    groups = list(grouped.values())
    rng.shuffle(groups)
    targets = [sum(len(items) for items in groups) * ratio / 100 for ratio in split]
    partitions = [[], [], []]
    counts = [0, 0, 0]
    for items in groups:
        target_index = min(
            range(3),
            key=lambda index: (counts[index] - targets[index], index),
        )
        partitions[target_index].extend(items)
        counts[target_index] += len(items)
    return tuple(partitions)


def _split_plan(train_records, validation_records, test_records, split, group_aware):
    """Persist the exact partition policy and counts used by a training run."""
    partitions = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }
    return {
        "strategy": "group-aware" if group_aware else "stratified",
        "ratios": list(split),
        "counts": {name: len(items) for name, items in partitions.items()},
        "classes": {
            name: sorted({record["label"] for record in items})
            for name, items in partitions.items()
        },
        "group_overlap": _group_overlap(partitions) if group_aware else [],
    }


def _group_overlap(partitions):
    overlaps = []
    names = list(partitions)
    for index, first_name in enumerate(names):
        first_groups = {
            record.get("group_id") for record in partitions[first_name]
            if record.get("group_id")
        }
        for second_name in names[index + 1:]:
            second_groups = {
                record.get("group_id") for record in partitions[second_name]
                if record.get("group_id")
            }
            overlaps.extend(
                f"{first_name}/{second_name}:{group}"
                for group in sorted(first_groups & second_groups)
            )
    return overlaps


def _read_images(dataset, records):
    """Read only validated archive members into normalized CHW tensors later."""
    with dataset.uploaded_file.open("rb") as upload:
        with zipfile.ZipFile(upload) as archive:
            samples = []
            for record in records:
                with archive.open(record["filename"]) as member:
                    samples.append((member.read(), record["label"]))
    return samples


def _build_model(torch, nn, architecture, class_count, model_version=3):
    widths = {
        "efficientnet_b0": (16, 32),
        "resnet50": (24, 48),
        "mobilenet_v3": (8, 16),
    }
    first_width, second_width = widths[architecture]
    if model_version == 1:
        return nn.Sequential(
            nn.Conv2d(3, first_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first_width, second_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(second_width, class_count),
        )
    if model_version >= 3:
        return nn.Sequential(
            nn.Conv2d(3, first_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(first_width, second_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(second_width, second_width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(second_width * 2, second_width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second_width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(second_width * 4, second_width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(second_width * 2, class_count),
        )
    return nn.Sequential(
        nn.Conv2d(3, first_width, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(first_width, second_width, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(second_width, second_width * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(second_width * 2 * 4 * 4, second_width * 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(second_width * 2, class_count),
    )


def _image_tensor(torch, image, image_size, normalize=True, mean=IMAGE_MEAN, std=IMAGE_STD):
    from PIL import Image

    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    values = torch.tensor(pixels, dtype=torch.float32).reshape(image_size, image_size, 3)
    tensor = values.permute(2, 0, 1).div(255.0)
    if normalize:
        mean_tensor = torch.tensor(tuple(mean), dtype=torch.float32).reshape(3, 1, 1)
        std_tensor = torch.tensor(tuple(std), dtype=torch.float32).reshape(3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor
    return tensor


def _tensor_dataset(torch, samples, label_to_index, augment=False):
    from PIL import Image

    images, labels = [], []
    for image_bytes, label in samples:
        with Image.open(io.BytesIO(image_bytes)) as image:
            tensor = _image_tensor(torch, image, IMAGE_SIZE)
        images.append(tensor)
        labels.append(label_to_index[label])
        if augment:
            images.append(torch.flip(tensor, dims=(2,)))
            labels.append(label_to_index[label])
    if not images:
        return None
    return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))


def _collect_outputs(torch, model, data_loader):
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


def _fit_temperature(torch, logits, targets):
    """Pick a deterministic temperature that minimizes held-out NLL."""
    if logits is None or targets is None or len(targets) < 2:
        return 1.0
    loss_function = torch.nn.CrossEntropyLoss()
    candidates = [0.5 + index * 0.05 for index in range(51)]
    losses = [
        float(loss_function(logits / temperature, targets).item())
        for temperature in candidates
    ]
    return round(candidates[min(range(len(losses)), key=losses.__getitem__)], 2)


def _evaluate(torch, model, data_loader, labels, class_count):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logits, actual_tensor = _collect_outputs(torch, model, data_loader)
    if logits is None:
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
    predictions = logits.argmax(dim=1).tolist()
    actual = actual_tensor.tolist()
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    class_labels = list(range(class_count))
    try:
        auroc = float(roc_auc_score(actual, probabilities, multi_class="ovr", labels=class_labels))
    except ValueError:
        auroc = None
    per_class = []
    class_f1 = f1_score(actual, predictions, labels=class_labels, average=None, zero_division=0)
    class_precision = precision_score(actual, predictions, labels=class_labels, average=None, zero_division=0)
    class_recall = recall_score(actual, predictions, labels=class_labels, average=None, zero_division=0)
    for index, label in enumerate(labels):
        per_class.append({
            "label": label,
            "precision": round(float(class_precision[index]), 4),
            "recall": round(float(class_recall[index]), 4),
            "f1": round(float(class_f1[index]), 4),
        })
    return {
        "accuracy": round(float(accuracy_score(actual, predictions)), 4),
        "macro_f1": round(float(f1_score(actual, predictions, labels=class_labels, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predictions)), 4),
        "precision": round(float(precision_score(actual, predictions, labels=class_labels, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predictions, labels=class_labels, average="macro", zero_division=0)), 4),
        "auroc": round(auroc, 4) if auroc is not None else None,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(actual, predictions, labels=class_labels).tolist(),
        "sample_count": len(actual),
    }


def _calibration_metrics(torch, logits, targets):
    if logits is None or targets is None or len(targets) == 0:
        return {"status": "insufficient_data", "ece": None, "brier": None, "bins": []}
    probabilities = torch.softmax(logits, dim=1).detach().cpu()
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets.cpu())
    bins = []
    ece = 0.0
    for lower in [index / 10 for index in range(10)]:
        upper = lower + 0.1
        selected = (confidences > lower) & (confidences <= upper if upper < 1 else confidences <= upper)
        if not selected.any():
            continue
        accuracy = float(correct[selected].float().mean())
        confidence = float(confidences[selected].mean())
        count = int(selected.sum())
        ece += abs(accuracy - confidence) * count / len(targets)
        bins.append({
            "range": f"{lower:.1f}–{upper:.1f}",
            "accuracy": round(accuracy, 4),
            "confidence": round(confidence, 4),
            "count": count,
        })
    one_hot = torch.nn.functional.one_hot(targets.cpu(), num_classes=probabilities.shape[1]).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean())
    return {
        "status": "completed",
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "bins": bins,
    }


def run_training(run, progress_callback=None):
    """Train and evaluate one experiment, returning persisted metric values."""
    progress_callback = progress_callback or getattr(run, "progress_callback", None)
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run a training experiment.") from exc

    # Keep CPU work bounded for the shared workspace while using more than
    # one core when it is available. This is substantially faster than the
    # previous single-threaded default without overwhelming the host.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.manual_seed(RANDOM_SEED)

    _report_progress(
        progress_callback,
        "validate",
        8,
        "Validating your dataset",
        "Checking image files, labels, and readable samples.",
    )
    with run.dataset.uploaded_file.open("rb") as dataset_file:
        report = validate_upload(dataset_file)
    if not report["valid"]:
        errors = "; ".join(report["errors"][:3]) or "Dataset validation failed."
        raise ValueError(errors)

    split = _normalize_split(run.config.get("split_ratios", DEFAULT_SPLIT))
    epochs = int(run.config.get("epochs", EPOCHS))
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    split_function = _group_split_records if any(record.get("group_id") for record in report["records"]) else _split_records
    train_records, validation_records, test_records = split_function(report["records"], split)
    group_aware = split_function is _group_split_records
    split_plan = _split_plan(
        train_records,
        validation_records,
        test_records,
        split,
        group_aware,
    )
    if split_plan["group_overlap"]:
        raise ValueError("The selected split contains overlapping entity groups.")
    classes = sorted(report["classes"])
    label_to_index = {label: index for index, label in enumerate(classes)}
    _report_progress(
        progress_callback,
        "load",
        16,
        "Loading image samples",
        f"Reading {len(report['records']):,} validated images from the archive.",
    )
    train_samples = _read_images(run.dataset, train_records)
    validation_samples = _read_images(run.dataset, validation_records)
    test_samples = _read_images(run.dataset, test_records)
    _report_progress(
        progress_callback,
        "prepare",
        27,
        "Preparing model inputs",
        f"Resizing images to {IMAGE_SIZE}×{IMAGE_SIZE} and applying normalization.",
    )
    train_data = _tensor_dataset(torch, train_samples, label_to_index, augment=True)
    validation_data = _tensor_dataset(torch, validation_samples, label_to_index)
    test_data = _tensor_dataset(torch, test_samples, label_to_index)
    if train_data is None:
        raise ValueError("The audited dataset has no readable training images.")

    batch_size = min(MAX_BATCH_SIZE, len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    validation_loader = (
        torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
        if validation_data else None
    )
    test_loader = (
        torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        if test_data else None
    )

    model = _build_model(torch, nn, run.architecture, len(classes), model_version=3)
    class_counts = [sum(1 for _, label in train_samples if label == class_name) for class_name in classes]
    weights = torch.tensor(
        [len(train_samples) / max(1, len(classes) * count) for count in class_counts],
        dtype=torch.float32,
    )
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_losses = []

    _report_progress(
        progress_callback,
        "train",
        34,
        "Training the model",
        f"Running {epochs} epoch{'s' if epochs != 1 else ''} on the CPU.",
    )
    model.train()
    for epoch in range(epochs):
        total_loss, batch_count = 0.0, 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        epoch_losses.append(round(total_loss / max(1, batch_count), 5))
        _report_progress(
            progress_callback,
            "train",
            34 + round(45 * (epoch + 1) / epochs),
            "Training the model",
            f"Epoch {epoch + 1} of {epochs} complete · loss {epoch_losses[-1]:.4f}.",
        )

    _report_progress(
        progress_callback,
        "evaluate",
        86,
        "Evaluating performance",
        "Calculating held-out accuracy and macro-F1.",
    )
    evaluation_loader = test_loader or validation_loader or train_loader
    evaluation_split = "test" if test_loader else "validation" if validation_loader else "training"
    evaluation = _evaluate(torch, model, evaluation_loader, classes, len(classes))
    evaluation_logits, evaluation_targets = _collect_outputs(torch, model, evaluation_loader)
    calibration_loader = validation_loader or test_loader
    calibration_logits, calibration_targets = _collect_outputs(torch, model, calibration_loader)
    temperature = _fit_temperature(torch, calibration_logits, calibration_targets)
    calibration = _calibration_metrics(
        torch,
        calibration_logits / temperature if calibration_logits is not None else None,
        calibration_targets,
    )
    calibration_report = build_calibration_report(calibration)
    evaluation_records = test_records or validation_records
    evaluation_predictions = (
        evaluation_logits.argmax(dim=1).tolist()
        if evaluation_logits is not None else []
    )
    fairness = build_fairness_report(
        evaluation_records,
        evaluation_targets.tolist() if evaluation_targets is not None else [],
        evaluation_predictions,
        classes,
    )
    statistics = build_run_statistics(evaluation)
    training_distribution = {
        label: count for label, count in zip(classes, class_counts)
    }
    experiment_variant = run.config.get("experiment_variant", "original")
    experiment_plan = build_experiment_plan(experiment_variant)

    _report_progress(
        progress_callback,
        "checkpoint",
        96,
        "Saving the trained checkpoint",
        "Packaging model weights and class metadata for prediction.",
    )
    checkpoint_dir = Path(settings.MEDIA_ROOT) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"training_run_{run.pk}.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "architecture": run.architecture,
            "model_version": 3,
            "classes": classes,
            "class_to_index": label_to_index,
            "num_classes": len(classes),
            "preprocessing": {
                "image_size": IMAGE_SIZE,
                "color_mode": "RGB",
                "normalize": True,
                "mean": list(IMAGE_MEAN),
                "std": list(IMAGE_STD),
            },
            # Horizontal flips are used as training augmentation only. They
            # are not averaged at inference because some generic datasets
            # contain orientation-sensitive classes.
            "inference": {"horizontal_flip_tta": False},
            "temperature": temperature,
            "state_dict": model.state_dict(),
            "evaluation": evaluation,
            "calibration": calibration,
            "fairness": fairness,
            "training_distribution": training_distribution,
            "dataset_id": run.dataset_id,
            "dataset_pipeline": run.dataset.pipeline if run.dataset else {},
        },
        checkpoint_path,
    )

    return {
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "epochs": epochs,
        "split": " / ".join(str(value) for value in split),
        "split_ratios": list(split),
        "evaluation_split": evaluation_split,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_samples": len(test_records),
        "classes": classes,
        "device": "CPU",
        "execution": "completed",
        "model_version": 3,
        "class_to_index": label_to_index,
        "preprocessing": {
            "image_size": IMAGE_SIZE,
            "color_mode": "RGB",
            "normalize": True,
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "temperature": temperature,
        "evaluation": evaluation,
        "calibration": calibration,
        "reliability": {
            "calibration": calibration_report,
            "fairness": fairness,
        },
        "statistics": statistics,
        "training_distribution": training_distribution,
        "split_plan": split_plan,
        "experiment_variant": experiment_variant,
        "experiment_plan": experiment_plan,
        "workflow_stages": {
            "stage_6_experiments": "completed",
            "stage_7_modeling": "completed",
            "stage_8_evaluation": "completed",
            "stage_9_reliability": "completed",
            "stage_10_xai": "available_on_prediction",
            "stage_11_model_store": "completed",
            "stage_12_monitoring": "available_after_predictions",
        },
        "checkpoint": str(checkpoint_path.relative_to(settings.MEDIA_ROOT)),
        "final_loss": epoch_losses[-1],
        "loss_history": epoch_losses,
    }
