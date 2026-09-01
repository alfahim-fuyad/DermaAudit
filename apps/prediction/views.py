from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from .models import Prediction
from .services.predictor import _latest_checkpoint, analyze_image
from apps.reports.services.cleanup import clear_predictions


def predict(request):
    if request.method == "POST":
        upload = request.FILES.get("image")
        if not upload:
            messages.error(request, "Choose an image before continuing.")
        else:
            try:
                analysis = analyze_image(upload)
                upload.seek(0)
                item = Prediction.objects.create(image=upload, predicted_class=analysis["label"],
                                                 confidence=analysis["confidence"],
                                                 review_required=analysis["review_required"],
                                                 explanation=analysis)
                return redirect("prediction:result", pk=item.pk)
            except Exception as exc:
                if isinstance(exc, (ValueError, OSError)):
                    messages.error(request, str(exc))
                else:
                    messages.error(request, "That file could not be read as an image. Use JPG, PNG, or WEBP.")
    active_model, _checkpoint_path = _latest_checkpoint()
    return render(
        request,
        "prediction/predict.html",
        {"page_title": "New prediction", "active_model": active_model},
    )


def result(request, pk):
    item = get_object_or_404(Prediction, pk=pk)
    return render(request, "prediction/result.html", {"prediction": item, "page_title": "Prediction result"})


def history(request):
    predictions = Prediction.objects.all()
    return render(
        request,
        "prediction/history.html",
        {"predictions": predictions, "page_title": "Prediction history"},
    )


def clear_history(request):
    if request.method != "POST":
        from django.http import HttpResponseNotAllowed
        return HttpResponseNotAllowed(["POST"])
    result = clear_predictions()
    messages.success(request, f"Cleared {result['predictions']} prediction record(s) from history.")
    return redirect("prediction:history")
