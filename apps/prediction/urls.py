from django.urls import path
from . import views

app_name = "prediction"
urlpatterns = [
    path("", views.predict, name="predict"),
    path("result/<int:pk>/", views.result, name="result"),
]
