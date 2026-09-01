from django.contrib import messages
from django.shortcuts import redirect, render
from apps.datasets.models import Dataset
from .models import TrainingRun
from .services.trainer import DEFAULT_SPLIT, EPOCHS, run_training


MAX_EPOCHS = 200


def _training_context(split_defaults=None, epochs_default=EPOCHS):
    split_defaults = split_defaults or {
        "train": DEFAULT_SPLIT[0],
        "validation": DEFAULT_SPLIT[1],
        "test": DEFAULT_SPLIT[2],
    }
    return {
        "datasets": Dataset.objects.filter(status="ready"),
        "runs": TrainingRun.objects.select_related("dataset")[:6],
        "split_defaults": split_defaults,
        "epochs_default": epochs_default,
        "page_title": "Model training",
    }


def training_home(request):
    return render(request, "training/training.html", _training_context())


def start_training(request):
    if request.method == "POST":
        dataset = Dataset.objects.filter(pk=request.POST.get("dataset"), status="ready").first()
        architecture = request.POST.get("architecture", "efficientnet_b0")
        valid_architectures = {value for value, _label in TrainingRun.ARCHITECTURES}
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
        if any(value < 1 for value in split) or sum(split) != 100:
            messages.error(request, "Training, validation, and test splits must be at least 1% and total 100%.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if not 1 <= epochs <= MAX_EPOCHS:
            messages.error(request, f"Epochs must be between 1 and {MAX_EPOCHS}.")
            return render(request, "training/training.html", _training_context(split_defaults, epochs_default))
        if architecture not in valid_architectures:
            messages.error(request, "Choose a supported model architecture.")
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
                "dataset_pipeline": dataset.pipeline,
            },
        )
        try:
            result = run_training(run)
        except Exception as exc:
            run.status = "failed"
            run.config = {
                **run.config,
                "execution": "failed",
                "error": str(exc),
            }
            run.save(update_fields=["status", "config"])
            messages.error(request, f"Training failed: {exc}")
        else:
            run.status = "completed"
            run.accuracy = result.pop("accuracy")
            run.macro_f1 = result.pop("macro_f1")
            run.config = {**run.config, **result}
            run.save(update_fields=["status", "accuracy", "macro_f1", "config"])
            messages.success(
                request,
                f"{run.get_architecture_display()} training completed "
                f"(accuracy {run.accuracy:.3f}, macro-F1 {run.macro_f1:.3f}).",
            )
    return redirect("training:home")


def experiments(request):
    runs = TrainingRun.objects.select_related("dataset")
    return render(request, "training/experiments.html", {"runs": runs, "page_title": "Experiments"})
