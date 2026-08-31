from django.contrib import messages
from django.shortcuts import redirect, render
from apps.datasets.models import Dataset
from .models import TrainingRun


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
                status="ready",
                config={
                    "epochs": 20,
                    "split": "70 / 15 / 15",
                    "device": "CPU preview",
                    "execution": "queued",
                    "dataset_pipeline": dataset.pipeline,
                },
            )
            messages.success(request, f"{run.get_architecture_display()} benchmark queued for training.")
    return redirect("training:home")


def experiments(request):
    runs = TrainingRun.objects.select_related("dataset")
    return render(request, "training/experiments.html", {"runs": runs, "page_title": "Experiments"})
