from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .models import Profile


class AuthenticationFlowTests(TestCase):
    def test_registration_creates_user_profile_and_logs_in(self):
        response = self.client.post(reverse("accounts:register"), {
            "name": "Dr. Alex Morgan",
            "email": "Alex@Example.org",
            "organization": "Research Lab",
            "password": "safe-password-123",
        })

        self.assertRedirects(response, reverse("accounts:home"))
        user = User.objects.get(username="alex@example.org")
        self.assertTrue(user.check_password("safe-password-123"))
        self.assertEqual(user.profile.organization, "Research Lab")
        self.assertEqual(self.client.session.get("_auth_user_id"), str(user.pk))

    def test_registration_rejects_short_password(self):
        response = self.client.post(reverse("accounts:register"), {
            "name": "Alex Morgan",
            "email": "alex@example.org",
            "password": "short",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Password must be at least 8 characters.")
        self.assertFalse(User.objects.filter(username="alex@example.org").exists())

    def test_duplicate_email_is_rejected_case_insensitively(self):
        User.objects.create_user(username="alex@example.org", email="alex@example.org", password="safe-password-123")

        response = self.client.post(reverse("accounts:register"), {
            "name": "Another Alex",
            "email": "ALEX@EXAMPLE.ORG",
            "password": "safe-password-123",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "An account with that email already exists.")
        self.assertEqual(User.objects.filter(username="alex@example.org").count(), 1)

    def test_login_accepts_normalized_email(self):
        user = User.objects.create_user(
            username="alex@example.org",
            email="alex@example.org",
            password="safe-password-123",
        )
        Profile.objects.create(user=user)

        response = self.client.post(reverse("accounts:login"), {
            "username": " ALEX@EXAMPLE.ORG ",
            "password": "safe-password-123",
        })

        self.assertRedirects(response, reverse("accounts:home"))
        self.assertEqual(self.client.session.get("_auth_user_id"), str(user.pk))

    def test_invalid_login_shows_error_without_authenticating(self):
        User.objects.create_user(username="alex@example.org", password="safe-password-123")

        response = self.client.post(reverse("accounts:login"), {
            "username": "alex@example.org",
            "password": "wrong-password",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "That email or password was not recognised.")
        self.assertNotIn("_auth_user_id", self.client.session)