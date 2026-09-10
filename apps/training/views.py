from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET

from apps.datasets.models import Dataset
from apps.reports.services.cleanup import clear_training_runs
from .models import TrainingRun
from .services.experiments import EXPERIMENT_VARIANTS, build_experiment_plan
from .services.trainer import DEFAULT_SPLIT, EPOCHS, run_training


TRAINING_WORKFLOW_STAGES = (
    ("stage_6_experiments", "Stage 6 · Experiments", "Compare controlled benchmark variants"),
    ("stage_7_modeling", "Stage 7 · Modeling", "Detect classes and configure the classification head"),
    ("stage_8_evaluation", "Stage 8 · Evaluation", "Measure held-out and per-class performance"),
    ("stage_9_reliability", "Stage 9 · Reliability", "Calibrate confidence and record reliability evidence"),
    ("stage_10_xai", "Stage 10 · XAI", "Generate Grad-CAM at prediction time"),
    ("stage_11_model_store", "Stage 11 · Model store", "Save versioned checkpoint and lineage"),
)


MAX_EPOCHS = 200
_TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dermaaudit-training")


def _training_context(split_defaults=None, epochs_default=EPOCHS):
    split_defaults = split_defaults or {
        "train": DEFAULT_SPLIT[0],
        "validation": DEFAULT_SPLIT[1],
        "test": DEFAULT_SPLIT[2],
    }
    return {
        "datasets": Dataset.objects.filter(status="ready"),
        "runs": TrainingRun.objects.select_related("dataset")[:6],
        "active_runs": TrainingRun.objects.filter(status="running").select_related("dataset"),
        "split_defaults": split_defaults,
        "epochs_default": epochs_default,
        "training_workflow_stages": TRAINING_WORKFLOW_STAGES,
        "experiment_variants": EXPERIMENT_VARIANTS,
        "page_title": "Model training",
    }


def training_home(request):
    return render(request, "training/training.html", _training_context())


def _complete_training(run_id, manage_connections=True):
    """Execute a queued run and persist its terminal state outside the request."""
    from django.db import close_old_connections

    if manage_connections:
        close_old_connections()
    run = None
    try:
        run = TrainingRun.objects.select_related("dataset").get(pk=run_id)
        def save_progress(progress):
            run.config = {**(run.config or {}), "progress": progress, "execution": "running"}
            run.save(update_fields=["config"])

        run.progress_callback = save_progress
        result = run_training(run)
    except Exception as exc:
        if run:
            run.status = "failed"
            run.config = {
                **(run.config or {}),
                "execution": "failed",
                "error": str(exc),
                "progress": {
                    "step": "failed",
                    "percent": 100,
                    "label": "Training needs attention",
                    "detail": str(exc),
                },
            }
            run.save(update_fields=["status", "config"])
    else:
        run.status = "completed"
        run.accuracy = result.pop("accuracy")
        run.macro_f1 = result.pop("macro_f1")
        run.config = {
            **(run.config or {}),
            **result,
            "execution": "completed",
            "progress": {
                "step": "complete",
                "percent": 100,
                "label": "Training complete",
                "detail": "Checkpoint saved and ready for prediction.",
            },
        }
        run.save(update_fields=["status", "accuracy", "macro_f1", "config"])
    finally:
        if manage_connections:
            close_old_connections()


def _safe_dataset_lookup(dataset_id):
    try:
        pk = int(dataset_id)
    except (TypeError, ValueError):
        return None
    return Dataset.objects.filter(pk=pk, status="ready").first()


def start_training(request):
    if request.method == "POST":
        dataset = _safe_dataset_lookup(request.POST.get("dataset"))
        architecture = request.POST.get("architecture", "efficientnet_b0")
        valid_architectures = {value for value, _label in TrainingRun.ARCHITECTURES}
        experiment_variant = request.POST.get("experiment_variant", "original")
        split_defaults = {
            "train": request.POST.get("train_split", DEFAULT_SPLIT[0]),
            "validation": request.POST.get("validation_split", DEFAULT_SPLIT[1]),
            "test": request.POST.get("test_split", DEFAULT_SPLIT[2]),
        }
        epochs_default = request.POST.get("epochs", EPOCHS)
        try:
            split = tuple(int(split_defaults[key]) for key in ("train", "validation", "test"))
            epochs = int(epochs_default)
        except (TypeError, ValueError):
            messages.error(request, "Split percentages and epochs must be whole numbers.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if not dataset:
            messages.error(request, "Choose a successfully audited dataset before training.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if dataset.training_runs.filter(status="running").exists():
            messages.error(request, "A training run is already active for this dataset.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if any(value < 1 for value in split) or sum(split) != 100:
            messages.error(request, "Training, validation, and test splits must be at least 1% and total 100%.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if not 1 <= epochs <= MAX_EPOCHS:
            messages.error(request, f"Epochs must be between 1 and {MAX_EPOCHS}.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if architecture not in valid_architectures:
            messages.error(request, "Choose a supported model architecture.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if experiment_variant not in {variant["key"] for variant in EXPERIMENT_VARIANTS}:
            messages.error(request, "Choose a supported experiment variant.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))

        run = TrainingRun.objects.create(
            dataset=dataset,
            architecture=architecture,
            status="running",
            config={
                "epochs": epochs,
                "split": " / ".join(str(value) for value in split),
                "split_ratios": list(split),
                "device": "CPU",
                "execution": "running",
                "experiment_variant": experiment_variant,
                "progress": {
                    "step": "validate",
                    "percent": 5,
                    "label": "Getting started",
                    "detail": "The training job is being prepared.",
                },
                "dataset_pipeline": dataset.pipeline,
            },
        )
        if settings.TESTING:
            _complete_training(run.pk, manage_connections=False)
        else:
            _TRAINING_EXECUTOR.submit(_complete_training, run.pk)
        messages.info(
            request,
            f"{run.get_architecture_display()} training started. "
            "You can stay on this page while the run completes.",
        )
    return redirect("training:home")


@require_GET
def training_status(request, pk):
    run = TrainingRun.objects.select_related("dataset").filter(pk=pk).first()
    if not run:
        return JsonResponse({"error": "Training run not found."}, status=404)
    return JsonResponse({
        "id": run.pk,
        "status": run.status,
        "status_label": run.get_status_display(),
        "accuracy": run.accuracy,
        "macro_f1": run.macro_f1,
        "error": run.config.get("error", ""),
        "execution": run.config.get("execution", ""),
        "progress": run.config.get("progress", {}),
        "is_active": run.is_active,
    })


def experiments(request):
    runs = TrainingRun.objects.select_related("dataset")
    return render(
        request,
        "training/experiments.html",
        {
            "runs": runs,
            "active_model": runs.filter(is_active=True, status="completed").first(),
            "experiment_plan": build_experiment_plan(),
            "page_title": "Experiments",
        },
    )


def _is_within_media(media_root, candidate):
    try:
        try:
            return candidate.is_relative_to(media_root)
        except AttributeError:
            return media_root in candidate.parents or candidate == media_root
    except (ValueError, RuntimeError):
        return False


def activate_model(request, pk):
    if request.method != "POST":
        from django.http import HttpResponseNotAllowed
        return HttpResponseNotAllowed(["POST"])

    run = get_object_or_404(TrainingRun.objects.select_related("dataset"), pk=pk)
    checkpoint_name = (run.config or {}).get("checkpoint")
    checkpoint_path = None
    if checkpoint_name:
        from pathlib import Path
        try:
            media_root = Path(settings.MEDIA_ROOT).resolve()
            checkpoint_path = (media_root / checkpoint_name).resolve()
        except (ValueError, RuntimeError, OSError):
            checkpoint_path = None
    else:
        media_root = None
    if (
        run.status != "completed"
        or not checkpoint_name
        or not checkpoint_path
        or not _is_within_media(media_root, checkpoint_path)
        or not checkpoint_path.is_file()
    ):
        messages.error(request, "Only a completed run with a saved checkpoint can be used for predictions.")
        return redirect("training:experiments")

    with transaction.atomic():
        TrainingRun.objects.filter(is_active=True).update(is_active=False)
        run.is_active = True
        run.save(update_fields=["is_active"])
    messages.success(
        request,
        f"{run.get_architecture_display()} is now selected for new predictions.",
    )
    return redirect("training:experiments")


def clear_experiments(request):
    if request.method != "POST":
        from django.http import HttpResponseNotAllowed
        return HttpResponseNotAllowed(["POST"])
    result = clear_training_runs()
    messages.success(request, f"Cleared {result['runs']} training record(s) from experiment history.")
    return redirect("training:experiments")
