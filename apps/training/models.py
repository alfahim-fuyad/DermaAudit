from django.db import models
from apps.datasets.models import Dataset


class TrainingRun(models.Model):
    ARCHITECTURES = [("efficientnet_b0", "EfficientNet-B0"), ("resnet50", "ResNet-50"), ("mobilenet_v3", "MobileNetV3")]
    STATUS = [("ready", "Ready"), ("running", "Running"), ("completed", "Completed"), ("failed", "Needs attention")]
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True, blank=True, related_name="training_runs")
    architecture = models.CharField(max_length=40, choices=ARCHITECTURES, default="efficientnet_b0")
    status = models.CharField(max_length=20, choices=STATUS, default="ready")
    accuracy = models.FloatField(default=0)
    macro_f1 = models.FloatField(default=0)
    config = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
