# DermaAudit AI

## Run

The application is a Django project. The Replit workflow runs:

```bash
python manage.py migrate && python manage.py runserver 0.0.0.0:5000
```

SQLite is used when no PostgreSQL connection variables are configured.

## Backend flow

1. Upload a ZIP containing labelled image folders or a recognised metadata file.
2. Validate archive safety, image readability, labels, duplicates, image quality, and metadata.
3. Build the dataset profile, canonical label mapping, split policy, leakage findings, bias availability, and imbalance recommendation.
4. Train a selected architecture using the persisted dataset configuration and deterministic CPU splits.
5. Store evaluation metrics, calibration, subgroup evidence when available, split details, experiment metadata, and a versioned checkpoint.
6. Activate a completed checkpoint for prediction. Prediction validates the image, applies the checkpoint, uses the confidence policy, and creates a Grad-CAM artifact when supported.
7. Reports expose workspace status, prediction review queue, class-distribution drift, and fairness evidence availability.

The product is for research and screening support only. It is not a medical diagnostic device.