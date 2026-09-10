from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from apps.datasets.models import Dataset
from apps.datasets.services.radar import build_radar_data
from apps.prediction.models import Prediction
from apps.prediction.services.predictor import _latest_checkpoint
from apps.training.models import TrainingRun
from .models import Profile


def home(request):
    current_time = timezone.localtime()
    if current_time.hour < 12:
        greeting = "Good morning"
    elif current_time.hour < 17:
        greeting = "Good afternoon"
    elif current_time.hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Good night"
    radar_data = build_radar_data(Dataset.objects.all())
    active_model, _checkpoint_path = _latest_checkpoint()
    model_registry = None
    if active_model:
        config = active_model.config or {}
        preprocessing = config.get("preprocessing") or {}
        model_registry = {
            "run": active_model,
            "accuracy_percent": round(float(active_model.accuracy or 0) * 100, 1),
            "macro_f1_percent": round(float(active_model.macro_f1 or 0) * 100, 1),
            "class_count": len(config.get("classes") or []),
            "epochs": config.get("epochs", "—"),
            "image_size": preprocessing.get("image_size", "—"),
        }
    datasets = list(Dataset.objects.all().order_by("-updated_at"))
    training_runs = list(TrainingRun.objects.select_related("dataset").order_by("-created_at"))
    predictions = list(Prediction.objects.all().order_by("-created_at"))
    completed_runs = [run for run in training_runs if run.status == "completed"]
    active_runs = [run for run in training_runs if run.status == "running"]
    failed_runs = [run for run in training_runs if run.status == "failed"]
    context = {
        "dataset_count": Dataset.objects.count(),
        "prediction_count": Prediction.objects.count(),
        "training_count": TrainingRun.objects.count(),
        "ready_dataset_count": sum(1 for dataset in datasets if dataset.status == "ready"),
        "active_training_count": len(active_runs),
        "training_completion": round(len(completed_runs) / len(training_runs) * 100) if training_runs else 0,
        "prediction_activity": min(100, Prediction.objects.count() * 10),
        "workspace_health": "Review needed" if failed_runs else "Ready",
        "workspace_health_detail": (
            f"{len(failed_runs)} training run{'s' if len(failed_runs) != 1 else ''} need attention"
            if failed_runs else "All systems operational"
        ),
        "active_training_runs": active_runs,
        "latest_prediction": predictions[0] if predictions else None,
        "radar_data": radar_data,
        "active_model": active_model,
        "model_registry": model_registry,
        "page_title": "Overview",
        "greeting": greeting,
        "current_time": current_time,
    }
    return render(request, "home/home.html", context)


def radar_status(request):
    """Return fresh dataset coverage for the live overview radar."""
    return JsonResponse(build_radar_data(Dataset.objects.all()))


def overview_status(request):
    """Return live workspace signals for the Overview pulse and activity feed."""
    datasets = list(Dataset.objects.all().order_by("-updated_at"))
    training_runs = list(TrainingRun.objects.select_related("dataset").order_by("-created_at"))
    predictions = list(Prediction.objects.all().order_by("-created_at"))
    completed_runs = [run for run in training_runs if run.status == "completed"]
    active_runs = [run for run in training_runs if run.status == "running"]
    failed_runs = [run for run in training_runs if run.status == "failed"]

    events = []
    for dataset in datasets:
        events.append({
            "kind": "Dataset",
            "title": dataset.name,
            "detail": f"{dataset.sample_count or 0:,} images · {dataset.get_status_display()}",
            "created_at": dataset.updated_at,
        })
    for run in training_runs:
        events.append({
            "kind": "Training",
            "title": run.get_architecture_display(),
            "detail": f"{run.dataset.name if run.dataset else 'Unassigned dataset'} · {run.get_status_display()}",
            "created_at": run.created_at,
        })
    for prediction in predictions:
        events.append({
            "kind": "Prediction",
            "title": prediction.predicted_class,
            "detail": f"{round(float(prediction.confidence or 0) * 100, 1)}% confidence",
            "created_at": prediction.created_at,
        })
    events.sort(key=lambda event: event["created_at"], reverse=True)

    return JsonResponse({
        "dataset_count": len(datasets),
        "ready_dataset_count": sum(1 for dataset in datasets if dataset.status == "ready"),
        "training_count": len(training_runs),
        "completed_training_count": len(completed_runs),
        "active_training_count": len(active_runs),
        "prediction_count": len(predictions),
        "training_completion": round(len(completed_runs) / len(training_runs) * 100) if training_runs else 0,
        "prediction_activity": min(100, len(predictions) * 10),
        "workspace_health": "Review needed" if failed_runs else "Ready",
        "workspace_health_detail": (
            f"{len(failed_runs)} training run{'s' if len(failed_runs) != 1 else ''} need attention"
            if failed_runs else "All systems operational"
        ),
        "active_runs": [
            {
                "label": run.get_architecture_display(),
                "dataset": run.dataset.name if run.dataset else "Unassigned dataset",
                "percent": run.config.get("progress", {}).get("percent", 0),
                "stage": run.config.get("progress", {}).get("label", "Training in progress"),
            }
            for run in active_runs
        ],
        "latest_prediction": {
            "label": predictions[0].predicted_class,
            "confidence": round(float(predictions[0].confidence or 0) * 100, 1),
        } if predictions else None,
        "activity": [
            {
                "kind": event["kind"],
                "title": event["title"],
                "detail": event["detail"],
                "when": timezone.localtime(event["created_at"]).isoformat(),
            }
            for event in events[:5]
        ],
        "updated_at": timezone.localtime().isoformat(),
    })


def login_view(request):
    if request.user.is_authenticated:
        return redirect("accounts:home")
    if request.method == "POST":
        username = request.POST.get("username", "").strip().lower()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("accounts:home")
        messages.error(request, "That email or password was not recognised.")
    return render(request, "accounts/login.html", {"page_title": "Sign in"})


def register_view(request):
    if request.user.is_authenticated:
        return redirect("accounts:home")
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        username = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        organization = request.POST.get("organization", "").strip()
        if not username or not password:
            messages.error(request, "Email and password are required.")
        elif len(password) < 8:
            messages.error(request, "Password must be at least 8 characters.")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "An account with that email already exists.")
        else:
            try:
                with transaction.atomic():
                    user = User.objects.create_user(
                        username=username,
                        email=username,
                        password=password,
                        first_name=name,
                    )
                    Profile.objects.create(user=user, organization=organization)
            except IntegrityError:
                messages.error(request, "An account with that email already exists.")
            else:
                login(request, user)
                return redirect("accounts:home")
    return render(request, "accounts/register.html", {"page_title": "Create account"})


@login_required
def profile_view(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)
    return render(request, "accounts/profile.html", {"profile": profile, "page_title": "Profile"})


@require_http_methods(["GET", "POST"])
def logout_view(request):
    """Allow both GET and POST for logout to support Django 5.2+ where LogoutView requires POST.

    The template uses a POST form, but we also handle GET for backwards compatibility
    and for cases where a user follows a direct logout link.
    """
    logout(request)
    messages.success(request, "You have been signed out.")
    return redirect("accounts:login")
