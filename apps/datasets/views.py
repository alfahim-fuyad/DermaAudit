from django.contrib import messages
from django.db import transaction
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
    return render(request, "datasets/audit.html", {"dataset": dataset, "audit": audit, "page_title": "Dataset audit"})
