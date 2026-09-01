from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.shortcuts import redirect, render
from django.utils import timezone
from apps.datasets.models import Dataset
from apps.prediction.models import Prediction
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
    context = {
        "dataset_count": Dataset.objects.count(),
        "prediction_count": Prediction.objects.count(),
        "training_count": TrainingRun.objects.count(),
        "active_model": "No model registered",
        "page_title": "Overview",
        "greeting": greeting,
        "current_time": current_time,
    }
    return render(request, "home/home.html", context)


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
