from django.shortcuts import render
from apps.datasets.models import Dataset
from apps.prediction.models import Prediction
from apps.training.models import TrainingRun


def dashboard(request):
    return render(request, "reports/dashboard.html", {
        "datasets": Dataset.objects.all()[:5],
        "runs": TrainingRun.objects.select_related("dataset")[:5],
        "predictions": Prediction.objects.all()[:5],
        "page_title": "Reports",
    })
