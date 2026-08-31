from django.contrib import messages
from django.db import transaction
from django.http import HttpResponseNotAllowed
from django.shortcuts import get_object_or_404, redirect, render
from .models import Dataset
from .services.pipeline import run_dataset_pipeline


def dataset_list(request):
    datasets = Dataset.objects.all()
    return render(request, "datasets/profile.html", {"datasets": datasets, "page_title": "Datasets"})


def dataset_upload(request):
    if request.method == "POST":
        upload = request.FILES.get("dataset_file")
        if not upload:
            messages.error(request, "Choose a ZIP dataset before continuing.")
            return render(request, "datasets/upload.html", {"page_title": "Upload dataset"})
        name = request.POST.get("name", "").strip() or (upload.name.rsplit(".", 1)[0] if upload else "Untitled dataset")
        with transaction.atomic():
            dataset = Dataset.objects.create(
                name=name[:180],
                uploaded_file=upload,
                status="auditing",
                pipeline={"status": "queued"},
            )
            run_dataset_pipeline(dataset)
        if dataset.status == "ready":
            messages.success(request, f"{dataset.name} was validated and profiled successfully.")
        else:
            messages.warning(request, f"{dataset.name} needs review before it can be used for training.")
        return redirect(dataset.get_absolute_url())
    return render(request, "datasets/upload.html", {"page_title": "Upload dataset"})


def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    return render(request, "datasets/details.html", {"dataset": dataset, "page_title": dataset.name})


def dataset_delete(request, pk):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    dataset = get_object_or_404(Dataset, pk=pk)
    dataset_name = dataset.name
    uploaded_file = dataset.uploaded_file
    dataset.delete()
    if uploaded_file:
        uploaded_file.delete(save=False)
    messages.success(request, f"{dataset_name} was deleted from your workspace.")
    return redirect("datasets:list")


def dataset_audit(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        run_dataset_pipeline(dataset)
        if dataset.status == "ready":
            messages.success(request, "Audit complete. The dataset configuration is ready for training.")
        else:
            messages.warning(request, "Audit complete. Review the validation findings before training.")
        return redirect("datasets:audit", pk=dataset.pk)
    audit = dataset.audit or {"duplicate_rate": "Pending", "quality_score": "—", "leakage_risk": "Not audited",
                              "label_consistency": "Pending", "imbalance_ratio": "Pending",
                              "recommendation": "Run an audit to generate recommendations."}
    stages = audit.get("stages", {})
    progress = {
        "profile": bool(dataset.profile),
        "quality": bool(audit.get("quality_score") not in (None, "—", "Pending")),
        "leakage": stages.get("leakage") == "completed",
        "ready": dataset.status == "ready" and stages.get("imbalance") == "completed",
    }
    completed_steps = sum(progress.values())
    progress_percent = {0: 0, 1: 0, 2: 33, 3: 66, 4: 100}[completed_steps]
    if progress["ready"]:
        current_step = "ready"
    elif progress["leakage"]:
        current_step = "ready"
    elif progress["quality"]:
        current_step = "leakage"
    elif progress["profile"]:
        current_step = "quality"
    else:
        current_step = "profile"
    return render(request, "datasets/audit.html", {
        "dataset": dataset,
        "audit": audit,
        "progress": progress,
        "current_step": current_step,
        "progress_percent": progress_percent,
        "page_title": "Dataset audit",
    })
