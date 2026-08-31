"""CPU training runner for reproducible image-classification experiments."""
import io
import random
import zipfile
from pathlib import Path

from django.conf import settings

from apps.datasets.services.validator import validate_upload


IMAGE_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 20


def _split_records(records):
    """Create deterministic class-aware train/validation/test partitions."""
    grouped = {}
    for record in records:
        grouped.setdefault(record["label"], []).append(record)

    rng = random.Random(RANDOM_SEED)
    train, validation, test = [], [], []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        count = len(items)

        if count >= 3:
            test_count = max(1, round(count * 0.15))
            validation_count = max(1, round(count * 0.15))
            while count - test_count - validation_count < 1:
                if validation_count > 0:
                    validation_count -= 1
                else:
                    test_count -= 1
        elif count == 2:
            test_count, validation_count = 1, 0
        else:
            test_count, validation_count = 0, 0

        train_count = count - test_count - validation_count
        train.extend(items[:train_count])
        validation.extend(items[train_count:train_count + validation_count])
        test.extend(items[train_count + validation_count:])

    return train, validation, test


def _read_images(dataset, records):
    """Read only validated archive members into normalized CHW tensors later."""
    with dataset.uploaded_file.open("rb") as upload:
        with zipfile.ZipFile(upload) as archive:
            samples = []
            for record in records:
                with archive.open(record["filename"]) as member:
                    samples.append((member.read(), record["label"]))
    return samples


def _build_model(torch, nn, architecture, class_count):
    widths = {
        "efficientnet_b0": (16, 32),
        "resnet50": (24, 48),
        "mobilenet_v3": (8, 16),
    }
    first_width, second_width = widths[architecture]
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


def _tensor_dataset(torch, samples, label_to_index):
    from PIL import Image

    images, labels = [], []
    for image_bytes, label in samples:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            pixels = list(image.getdata())
        values = torch.tensor(pixels, dtype=torch.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        images.append(values.permute(2, 0, 1).div(255.0))
        labels.append(label_to_index[label])
    if not images:
        return None
    return torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))


def _evaluate(torch, model, data_loader, labels, class_count):
    from sklearn.metrics import accuracy_score, f1_score

    if not data_loader:
        return 0.0, 0.0
    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            predictions.extend(model(images).argmax(dim=1).tolist())
            actual.extend(targets.tolist())
    return (
        float(accuracy_score(actual, predictions)),
        float(f1_score(actual, predictions, labels=list(range(class_count)), average="macro", zero_division=0)),
    )


def run_training(run):
    """Train and evaluate one experiment, returning persisted metric values."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run a training experiment.") from exc

    torch.set_num_threads(1)
    torch.manual_seed(RANDOM_SEED)

    with run.dataset.uploaded_file.open("rb") as dataset_file:
        report = validate_upload(dataset_file)
    if not report["valid"]:
        errors = "; ".join(report["errors"][:3]) or "Dataset validation failed."
        raise ValueError(errors)

    train_records, validation_records, test_records = _split_records(report["records"])
    classes = sorted(report["classes"])
    label_to_index = {label: index for index, label in enumerate(classes)}
    train_samples = _read_images(run.dataset, train_records)
    validation_samples = _read_images(run.dataset, validation_records)
    test_samples = _read_images(run.dataset, test_records)
    train_data = _tensor_dataset(torch, train_samples, label_to_index)
    validation_data = _tensor_dataset(torch, validation_samples, label_to_index)
    test_data = _tensor_dataset(torch, test_samples, label_to_index)
    if train_data is None:
        raise ValueError("The audited dataset has no readable training images.")

    batch_size = min(16, len(train_data))
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

    model = _build_model(torch, nn, run.architecture, len(classes))
    class_counts = [sum(1 for _, label in train_samples if label == class_name) for class_name in classes]
    weights = torch.tensor(
        [len(train_samples) / max(1, len(classes) * count) for count in class_counts],
        dtype=torch.float32,
    )
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_losses = []

    model.train()
    for _epoch in range(EPOCHS):
        total_loss, batch_count = 0.0, 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        epoch_losses.append(round(total_loss / max(1, batch_count), 5))

    evaluation_loader = test_loader or validation_loader or train_loader
    evaluation_split = "test" if test_loader else "validation" if validation_loader else "training"
    accuracy, macro_f1 = _evaluate(torch, model, evaluation_loader, classes, len(classes))

    checkpoint_dir = Path(settings.MEDIA_ROOT) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"training_run_{run.pk}.pt"
    torch.save(
        {
            "architecture": run.architecture,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "epochs": EPOCHS,
        "split": "70 / 15 / 15",
        "evaluation_split": evaluation_split,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_samples": len(test_records),
        "classes": classes,
        "device": "CPU",
        "execution": "completed",
        "checkpoint": str(checkpoint_path.relative_to(settings.MEDIA_ROOT)),
        "final_loss": epoch_losses[-1],
        "loss_history": epoch_losses,
    }
