from django.db import models
from django.urls import reverse


class Dataset(models.Model):
    STATUS_CHOICES = [
        ("uploaded", "Uploaded"),
        ("auditing", "Auditing"),
        ("ready", "Ready"),
        ("needs_review", "Needs review"),
    ]
    name = models.CharField(max_length=180)
    uploaded_file = models.FileField(upload_to="datasets/", blank=True, null=True)
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="uploaded")
    sample_count = models.PositiveIntegerField(default=0)
    class_count = models.PositiveIntegerField(default=0)
    classes = models.JSONField(default=list, blank=True)
    profile = models.JSONField(default=dict, blank=True)
    audit = models.JSONField(default=dict, blank=True)
    validation = models.JSONField(default=dict, blank=True)
    pipeline = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("datasets:detail", args=[self.pk])
