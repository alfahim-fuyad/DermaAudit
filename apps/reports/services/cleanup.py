"""Destructive workspace cleanup helpers with safe media-file removal."""
from pathlib import Path

from django.conf import settings

from apps.datasets.models import Dataset
from apps.prediction.models import Prediction
from apps.training.models import TrainingRun


def _safe_media_path(relative_name):
    if not relative_name:
        return None
    media_root = Path(settings.MEDIA_ROOT).resolve()
    candidate = (media_root / str(relative_name)).resolve()
    if media_root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def _remove_files(paths):
    removed = 0
    for path in paths:
        if path and path.is_file():
            path.unlink()
            removed += 1
    return removed


def _checkpoint_paths(runs):
    return {
        _safe_media_path((run.config or {}).get("checkpoint"))
        for run in runs
        if (run.config or {}).get("checkpoint")
    }


def delete_dataset(dataset):
    related_runs = list(TrainingRun.objects.filter(dataset=dataset))
    upload_path = _safe_media_path(dataset.uploaded_file.name) if dataset.uploaded_file else None
    checkpoint_paths = _checkpoint_paths(related_runs)
    dataset.delete()
    return {
        "datasets": 1,
        "runs": len(related_runs),
        "files": _remove_files({upload_path} | checkpoint_paths),
    }


def clear_datasets():
    datasets = list(Dataset.objects.all())
    related_runs = list(TrainingRun.objects.filter(dataset__isnull=False))
    upload_paths = {_safe_media_path(dataset.uploaded_file.name) for dataset in datasets if dataset.uploaded_file}
    checkpoint_paths = _checkpoint_paths(related_runs)
    dataset_count = len(datasets)
    run_count = len(related_runs)
    Dataset.objects.all().delete()
    return {
        "datasets": dataset_count,
        "runs": run_count,
        "files": _remove_files(upload_paths | checkpoint_paths),
    }


def clear_training_runs():
    runs = list(TrainingRun.objects.all())
    checkpoint_paths = _checkpoint_paths(runs)
    run_count = len(runs)
    TrainingRun.objects.all().delete()
    return {"runs": run_count, "files": _remove_files(checkpoint_paths)}


def clear_predictions():
    predictions = list(Prediction.objects.all())
    image_paths = {
        _safe_media_path(prediction.image.name)
        for prediction in predictions
        if prediction.image
    }
    prediction_count = len(predictions)
    Prediction.objects.all().delete()
    return {"predictions": prediction_count, "files": _remove_files(image_paths)}


def clear_workspace():
    datasets = clear_datasets()
    runs = clear_training_runs()
    predictions = clear_predictions()
    return {
        "datasets": datasets["datasets"],
        "runs": datasets["runs"] + runs["runs"],
        "predictions": predictions["predictions"],
        "files": datasets["files"] + runs["files"] + predictions["files"],
    }