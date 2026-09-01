from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainingrun",
            name="is_active",
            field=models.BooleanField(
                default=False,
                help_text="Use this completed checkpoint for new predictions.",
            ),
        ),
    ]