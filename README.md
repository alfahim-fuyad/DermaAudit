# DermaAudit AI

DermaAudit AI is a server-rendered Django workspace for skin-lesion image classification research. It keeps dataset validation, audit signals, model comparison, reports, and single-image review in one integrated application.

## Run on Replit or locally

```bash
python manage.py migrate --noinput
python manage.py runserver 0.0.0.0:5000
```

The Replit workflow **Start application** runs the migrations and starts Django on port `5000`, so the Preview panel can load the project immediately. The same command is suitable for local development. For a production-style process, use:

```bash
python manage.py migrate --noinput
gunicorn --bind 0.0.0.0:5000 config.wsgi:application
```

The project uses SQLite when `DATABASE_URL` is not set and PostgreSQL when it is. `SESSION_SECRET` is used for Django’s signing key when available; the development fallback should not be used for a public deployment.

## Research workflow

The application follows the attached evidence-first workflow:

1. Upload a compatible labeled image dataset.
2. Validate format, images, labels, structure, and classification assumptions.
3. Automatically profile image, label, metadata, class, and group information.
4. Harmonize labels and metadata, then audit quality, duplicates, imbalance, and leakage.
5. Use group-aware splits when entity IDs are available; otherwise use stratified splits.
6. Configure and run a reproducible training experiment with a held-out test set.
7. Compare model metrics, calibration signals, and per-class results in Reports.
8. Store a selected checkpoint in the model registry.
9. Upload one image for model-assisted prediction with confidence, alternatives, and Grad-CAM when available.
10. Monitor prediction volume, abstentions, review queue, and drift prerequisites.

Upload datasets as a single `.zip` with one folder per class, such as `benign/` and `malignant/`. A split dataset may use `train/`, `validation/`, and `test/` folders. Flat images can use a CSV/JSON manifest with image and label columns.

## Scope and model status

The interface is designed for research and screening support only; it is not a medical diagnostic device. The prediction screen validates images and provides a clearly labelled preview analysis until a trained checkpoint is registered for inference. Training experiments now run a deterministic CPU image classifier over the audited ZIP, report held-out metrics, and save a checkpoint under `media/checkpoints/`.

## Verification

Run the Django system check and test suite before handing off changes:

```bash
python manage.py check
python manage.py test
```

## Technology

Python, Django Templates, HTML5, CSS3, vanilla JavaScript, Pillow, NumPy, Pandas, scikit-learn, PyTorch/Torchvision, and PostgreSQL/Neon.
