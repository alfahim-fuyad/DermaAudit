from django.contrib import messages
from django.db import transaction
from django.http import HttpResponseNotAllowed
from django.shortcuts import get_object_or_404, redirect, render
from .models import Dataset
from .services.pipeline import DATASET_WORKFLOW_STAGES, run_dataset_pipeline
from .services.validator import MAX_ARCHIVE_BYTES
from apps.reports.services.cleanup import clear_datasets, delete_dataset


def dataset_list(request):
    datasets = Dataset.objects.all()
    return render(request, "datasets/profile.html", {"datasets": datasets, "page_title": "Datasets"})


def dataset_upload(request):
    if request.method == "POST":
        upload = request.FILES.get("dataset_file")
        if not upload:
            messages.error(request, "Choose a ZIP dataset before continuing.")
            return render(request, "datasets/upload.html", {"page_title": "Upload dataset"})
        if not upload.name.lower().endswith(".zip"):
            messages.error(request, "Upload a ZIP dataset. Other file types are not supported.")
            return render(request, "datasets/upload.html", {"page_title": "Upload dataset"})
        if upload.size > MAX_ARCHIVE_BYTES:
            messages.error(request, "The dataset exceeds the 7 GB upload limit.")
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


def dataset_clear(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    result = clear_datasets()
    messages.success(
        request,
        f"Cleared {result['datasets']} dataset(s) and {result['runs']} related training record(s).",
    )
    return redirect("datasets:list")


def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    return render(request, "datasets/details.html", {"dataset": dataset, "page_title": dataset.name})


def dataset_delete(request, pk):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    dataset = get_object_or_404(Dataset, pk=pk)
    dataset_name = dataset.name
    delete_dataset(dataset)
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
    cleaning = audit.get("cleaning") or None
    cleaning_class_rows = None
    if cleaning:
        counts_before = cleaning.get("class_counts_before") or {}
        counts_after = cleaning.get("class_counts_after") or {}
        max_count = max([*counts_before.values(), *counts_after.values(), 1])
        cleaning_class_rows = [
            {
                "label": label,
                "before": count,
                "after": counts_after.get(label, 0),
                "before_percent": round((count / max_count) * 100),
                "after_percent": round((counts_after.get(label, 0) / max_count) * 100),
            }
            for label, count in sorted(counts_before.items())
        ]
    stages = audit.get("stages", {})
    pipeline_workflow = (dataset.pipeline or {}).get("workflow", {})
    has_stage = lambda key, fallback: (
        pipeline_workflow.get(key, {}).get("status")
        or stages.get(fallback)
        or ("completed" if dataset.status == "ready" and dataset.profile else "")
    )
    progress = {
        "profile": bool(dataset.profile) or has_stage("stage_1", "profiling") == "completed",
        "quality": bool(audit.get("quality_score") not in (None, "—", "Pending")),
        "leakage": has_stage("stage_4", "leakage") == "completed",
        "cleaned": bool(
            (audit.get("cleaning") or {}).get("status") == "completed"
            or has_stage("cleaning", "cleaning") in {"completed", "passed"}
        ),
        "ready": dataset.status == "ready" and (
            has_stage("recheck", "recheck") in {"completed", "passed"}
            or has_stage("stage_5", "stage_5") == "completed"
            or stages.get("imbalance") == "completed"
        ),
    }
    completed_steps = sum(progress.values())
    # Fixed mapping: 0->0 … 5->100 for accurate progress bar
    progress_percent = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80, 5: 100}.get(completed_steps, 0)
    if progress["ready"]:
        current_step = "ready"
    elif progress["cleaned"]:
        current_step = "ready"
    elif progress["leakage"]:
        current_step = "cleaned"
    elif progress["quality"]:
        current_step = "leakage"
    elif progress["profile"]:
        current_step = "quality"
    else:
        current_step = "profile"
    return render(request, "datasets/audit.html", {
        "dataset": dataset,
        "audit": audit,
        "cleaning": cleaning,
        "cleaning_class_rows": cleaning_class_rows,
        "progress": progress,
        "current_step": current_step,
        "progress_percent": progress_percent,
        "workflow_stages": [
            {
                "key": key,
                "label": label,
                "description": description,
                 "status": pipeline_workflow.get(key, {}).get(
                    "status",
                    "completed" if stages.get(
                         {"stage_0": "validation", "stage_1": "profiling",
                         "stage_2": "harmonization", "stage_3": "audit",
                          "stage_4": "leakage", "stage_5": "stage_5",
                          "cleaning": "cleaning", "recheck": "recheck"}.get(key, key)
                    ) == "completed" else "pending",
                ),
                 "summary": pipeline_workflow.get(key, {}).get(
                    "summary", "Run the audit to generate this evidence."
                ),
            }
            for key, label, description in DATASET_WORKFLOW_STAGES
        ],
        "page_title": "Dataset audit",
    })
