from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    path("", views.home, name="home"),
    path("radar-status/", views.radar_status, name="radar_status"),
    path("overview-status/", views.overview_status, name="overview_status"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("profile/", views.profile_view, name="profile"),
]
