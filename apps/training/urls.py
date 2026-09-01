from django.urls import path
from . import views

app_name = "training"
urlpatterns = [
    path("", views.training_home, name="home"),
    path("start/", views.start_training, name="start"),
    path("status/<int:pk>/", views.training_status, name="status"),
    path("experiments/", views.experiments, name="experiments"),
    path("experiments/<int:pk>/activate/", views.activate_model, name="activate_model"),
    path("experiments/clear/", views.clear_experiments, name="clear_experiments"),
]
