from django.urls import path
from . import views

app_name = "prediction"
urlpatterns = [
    path("", views.predict, name="predict"),
    path("history/", views.history, name="history"),
    path("history/clear/", views.clear_history, name="clear_history"),
    path("result/<int:pk>/", views.result, name="result"),
]
