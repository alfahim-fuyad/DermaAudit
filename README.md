# DermaAudit AI

DermaAudit AI is a server-rendered Django workspace for skin-lesion image classification research. It keeps dataset validation, audit signals, model comparison, reports, and single-image review in one integrated application.

## Run locally

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:5000
```

The project uses SQLite when `DATABASE_URL` is not set and PostgreSQL/Neon when it is. Uploaded datasets belong in a ZIP with one folder per class, such as `benign/` and `malignant/`.

## Scope and model status

The interface is designed for research and screening support only; it is not a medical diagnostic device. The prediction screen validates images and provides a clearly labelled preview analysis until a trained Torch/Torchvision checkpoint is registered. The training workspace records reproducible benchmark configurations and is ready for a production checkpoint runner.

## Technology

Python, Django Templates, HTML5, CSS3, vanilla JavaScript, Pillow, NumPy, Pandas, scikit-learn, PyTorch/Torchvision, and PostgreSQL/Neon.
