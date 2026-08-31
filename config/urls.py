from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from apps.accounts.views import home

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("apps.accounts.urls")),
    path("datasets/", include("apps.datasets.urls")),
    path("training/", include("apps.training.urls")),
    path("prediction/", include("apps.prediction.urls")),
    path("reports/", include("apps.reports.urls")),
    path("", home, name="home"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
