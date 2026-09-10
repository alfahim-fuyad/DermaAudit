# DermaAudit AI

## Run locally or on Replit

Install the Python dependencies from `requirements.txt`, then run:

```bash
python manage.py migrate --noinput
python manage.py runserver 0.0.0.0:5000
```

The `Start application` workflow runs the migration step and serves the Django app on port `5000`.

The project uses SQLite when `DATABASE_URL` is not configured. `SESSION_SECRET` is used as Django's signing key when available; configure it before any public deployment.

## Project structure

- `apps/datasets`: generic labeled-image ingestion, auditing, cleaning, and validation
- `apps/training`: preprocessing, splitting, experiments, and model checkpoints
- `apps/prediction`: HAM10000-focused image prediction and review flow
- `apps/reports`: audit, evaluation, and monitoring views
- `templates` and `static`: server-rendered UI and interactions