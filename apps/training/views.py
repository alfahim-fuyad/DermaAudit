from django.contrib import messages
from django.shortcuts import redirect, render
from apps.datasets.models import Dataset
from .models import TrainingRun
from .services.trainer import run_training


def training_home(request):
    datasets = Dataset.objects.filter(status="ready")
    runs = TrainingRun.objects.select_related("dataset")[:6]
    return render(request, "training/training.html", {"datasets": datasets, "runs": runs, "page_title": "Model training"})


def start_training(request):
    if request.method == "POST":
        dataset = Dataset.objects.filter(pk=request.POST.get("dataset"), status="ready").first()
        architecture = request.POST.get("architecture", "efficientnet_b0")
        valid_architectures = {value for value, _label in TrainingRun.ARCHITECTURES}
        if not dataset:
            messages.error(request, "Choose a successfully audited dataset before training.")
        elif architecture not in valid_architectures:
            messages.error(request, "Choose a supported model architecture.")
        else:
            run = TrainingRun.objects.create(
                dataset=dataset,
                architecture=architecture,
                status="running",
                config={
                    "epochs": 20,
                    "split": "70 / 15 / 15",
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
