from django.contrib import admin
from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "sample_count", "class_count", "updated_at")
    list_filter = ("status", "created_at")
    search_fields = ("name",)
    readonly_fields = ("sample_count", "class_count", "classes", "profile", "validation", "audit", "pipeline")
