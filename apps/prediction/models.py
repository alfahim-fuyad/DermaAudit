from django.db import models


class Prediction(models.Model):
    image = models.ImageField(upload_to="predictions/")
    predicted_class = models.CharField(max_length=120, default="Awaiting analysis")
    confidence = models.FloatField(default=0)
    review_required = models.BooleanField(default=True)
    explanation = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
