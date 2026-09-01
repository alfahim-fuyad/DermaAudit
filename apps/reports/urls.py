from django.urls import path
from . import views

app_name = "reports"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("live-status/", views.live_status, name="live_status"),
    path("clear/", views.clear_all, name="clear_all"),
]
