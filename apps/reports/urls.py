from django.urls import path
from . import views

app_name = "reports"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("clear/", views.clear_all, name="clear_all"),
]
