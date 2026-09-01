from django.shortcuts import redirect, render
from django.contrib import messages
from django.http import HttpResponseNotAllowed, JsonResponse
from django.utils import timezone
from apps.datasets.models import Dataset
from apps.prediction.models import Prediction
from apps.training.models import TrainingRun
from .services.cleanup import clear_workspace


def _percent(value):
    try:
        return max(0, min(100, round(float(value) * 100, 1)))
    except (TypeError, ValueError):
        return 0


def dashboard(request):
    datasets = list(Dataset.objects.all().order_by("-updated_at"))
    runs = list(TrainingRun.objects.select_related("dataset").order_by("-created_at"))
    predictions = list(Prediction.objects.all().order_by("-created_at"))

    completed_runs = [run for run in runs if run.status == "completed"]
    active_model = next((run for run in completed_runs if run.is_active), None)
    audited_count = sum(1 for dataset in datasets if dataset.audit)
    ready_count = sum(1 for dataset in datasets if dataset.status == "ready")
    performance_chart = []
    for run in reversed(completed_runs[:6]):
        accuracy = _percent(run.accuracy)
        macro_f1 = _percent(run.macro_f1)
        performance_chart.append({
            "label": run.get_architecture_display().replace("EfficientNet-B0", "EfficientNet"),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "accuracy_height": max(2, accuracy),
            "macro_f1_height": max(2, macro_f1),
        })

    best_run = max(completed_runs, key=lambda run: run.accuracy, default=None)
    best_accuracy = _percent(best_run.accuracy) if best_run else None
    average_accuracy = (
        round(sum(run.accuracy for run in completed_runs) / len(completed_runs) * 100, 1)
        if completed_runs else 0
    )
    audit_coverage = round(audited_count / len(datasets) * 100) if datasets else 0
    readiness = round(ready_count / len(datasets) * 100) if datasets else 0

    if not datasets:
        next_action = "Upload a labelled dataset to start building evidence."
    elif not completed_runs:
        next_action = "Run a training experiment on a ready dataset to unlock model reporting."
    elif audit_coverage < 100:
        next_action = "Review the remaining dataset audit before comparing model performance."
    else:
        next_action = "Compare completed runs and use the strongest model for a new prediction."

    return render(request, "reports/dashboard.html", {
        "datasets": datasets,
        "runs": runs,
        "predictions": predictions,
        "recent_runs": runs[:5],
        "recent_datasets": datasets[:4],
        "performance_chart": performance_chart,
        "completed_count": len(completed_runs),
        "audited_count": audited_count,
        "ready_count": ready_count,
        "audit_coverage": audit_coverage,
        "readiness": readiness,
        "average_accuracy": average_accuracy,
        "best_run": best_run,
        "best_accuracy": best_accuracy,
        "active_model": active_model,
        "active_runs": [run for run in runs if run.status == "running"],
        "next_action": next_action,
        "page_title": "Reports",
    })


def live_status(request):
    """Return the small, safe-to-poll summary used by the live reports view."""
    datasets = list(Dataset.objects.all())
    runs = list(TrainingRun.objects.select_related("dataset").order_by("-created_at"))
    predictions_count = Prediction.objects.count()
    completed_runs = [run for run in runs if run.status == "completed"]
    active_runs = [run for run in runs if run.status == "running"]
    audited_count = sum(1 for dataset in datasets if dataset.audit)
    ready_count = sum(1 for dataset in datasets if dataset.status == "ready")
    best_run = max(completed_runs, key=lambda run: run.accuracy, default=None)
    active_model = next((run for run in completed_runs if run.is_active), None)
    return JsonResponse({
        "datasets": len(datasets),
        "runs": len(runs),
        "completed": len(completed_runs),
        "predictions": predictions_count,
        "audited": audited_count,
        "ready": ready_count,
        "best_accuracy": _percent(best_run.accuracy) if best_run else None,
        "best_model": best_run.get_architecture_display() if best_run else "",
        "active_model": {
            "id": active_model.pk,
            "label": active_model.get_architecture_display(),
            "dataset": active_model.dataset.name if active_model.dataset else "Unassigned dataset",
        } if active_model else None,
        "active_runs": [
            {
                "id": run.pk,
                "label": run.get_architecture_display(),
                "dataset": run.dataset.name if run.dataset else "Unassigned dataset",
                "progress": run.config.get("progress", {}),
            }
            for run in active_runs
        ],
        "updated_at": timezone.localtime().isoformat(),
    })


def clear_all(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    result = clear_workspace()
    messages.success(
        request,
        "Cleared "
        f"{result['datasets']} dataset(s), {result['runs']} training record(s), "
        f"and {result['predictions']} prediction record(s).",
    )
    return redirect("reports:dashboard")
