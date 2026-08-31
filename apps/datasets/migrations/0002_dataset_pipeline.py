from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("datasets", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="dataset",
            name="pipeline",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="dataset",
            name="validation",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]