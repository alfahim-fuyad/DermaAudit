# DermaAudit AI

DermaAudit AI is a server-rendered Django workspace for **generic image-classification dataset auditing and training**. Skin-lesion research is the flagship use case, but the same evidence-first workflow can be used for plant disease, X-ray, animal, food, or other labeled image datasets.

The platform is designed to make model decisions traceable:

```text
ZIP dataset → validate → profile → harmonize → audit
           → leakage-safe split → experiment → evaluate
           → calibrate → register checkpoint → predict → monitor
```

It is a research and screening support tool, not a medical diagnostic device.

## Run on Replit or locally

Install the declared Python dependencies, then run:

```bash
python manage.py migrate --noinput
python manage.py runserver 0.0.0.0:5000
```

The Replit workflow **Start application** runs the migrations and starts Django on port `5000`, so the Preview panel can load the project immediately. For a production-style process, use:

```bash
python manage.py migrate --noinput
gunicorn --bind 0.0.0.0:5000 config.wsgi:application
```

The project uses SQLite when `DATABASE_URL` is not set and PostgreSQL when it is. `SESSION_SECRET` is used for Django’s signing key when available; the development fallback should not be used for a public deployment.

## How the workflow works

### 1. Upload and validate

Upload one `.zip` archive from **Datasets**. The archive may contain:

- One folder per class, such as `benign/` and `malignant/`
- Split folders such as `train/`, `validation/`, and `test/`, with class folders inside
- Flat images paired with a recognized CSV or JSON manifest containing image and label columns

Before training, the validator checks supported formats, archive safety, readable images, corrupted files, dimensions, color modes, missing labels, class count, duplicate filenames, exact duplicate content, and near-duplicate content. The original archive is never modified.

### 2. Profile, harmonize, and audit

The dataset profile records image formats, resolutions, channels, class distribution, metadata files and columns, available IDs, group information, and subgroup fields. Labels are normalized without replacing the original upload.

The audit combines:

- Image quality and validation findings
- Exact and perceptual duplicate signals
- Label and metadata consistency
- Class imbalance and weighted-training recommendations
- Group/entity identifiers for leakage control
- Bias and fairness readiness when approved subgroup metadata exists

Datasets with blocking validation errors remain **Needs review** and cannot be used for training.

### 3. Choose a safe split and experiment

The default split is **70% train / 15% validation / 15% test**. Training enforces that the values total 100%.

- When matched patient, lesion, subject, case, study, or entity IDs exist, records are kept together with a group-aware split.
- Without group IDs, the pipeline uses a deterministic class-aware stratified split.
- Augmentation is applied only to training inputs; validation and test inputs remain untouched.

Each experiment can compare a declared intervention:

| Intervention | Training candidate |
| --- | --- |
| Original | Use the validated upload as-is |
| Duplicate-free | Exclude exact duplicate images |
| Leakage-controlled | Use group-aware splitting when group IDs exist |
| Bias-mitigated | Preserve subgroup evidence for review |
| Quality-controlled | Exclude low-resolution images |
| Imbalance-handled | Use class-weighted loss |
| Fully audited | Apply duplicate, near-duplicate, quality, group, and imbalance safeguards |

### 4. Train and evaluate

Choose one of the supported architecture profiles—EfficientNet-B0, ResNet-50, or MobileNetV3—and start an experiment from **Model training**. The CPU training runner records:

- Train loss and train Macro-F1 per epoch
- Validation Macro-F1 and learning rate per epoch
- AdamW optimization and learning-rate scheduling
- Early stopping and the best validation epoch
- Held-out accuracy, Macro-F1, balanced accuracy, precision, recall, AUROC, confusion matrix, and per-class metrics

The complete run record is available from **Reports → Model report**. Model selection should consider Macro-F1 and minority-class behavior, not accuracy alone.

### 5. Reliability, registry, and prediction

Each completed run saves a versioned checkpoint with its class mapping, preprocessing settings, split plan, intervention, evaluation metrics, calibration evidence, and dataset lineage. Select one checkpoint from **Experiments** before requesting a prediction.

The prediction review:

- Validates the uploaded image before inference
- Applies calibrated confidence when calibration data exists
- Shows the predicted class and top alternatives
- Abstains when confidence or top-class margin is too low
- Generates a Grad-CAM overlay when the checkpoint supports it
- Always recommends qualified human review

### 6. Monitoring

**Reports** provides live workspace signals for dataset readiness, training activity, prediction volume, abstentions, average confidence, and review queue size. Data drift is compared with the active model’s training class distribution only when both inputs exist.

Fairness drift and performance-after-deployment analysis remain explicitly unavailable until approved subgroup metadata and verified labels are available. Predictions are not treated as ground truth.

## Upload limits and supported images

- ZIP archives up to 2 GB
- Up to 10,000 archive files
- Images up to 50 MB each
- CSV/JSON metadata up to 10 MB per file
- JPG, JPEG, JFIF, PNG, WEBP, BMP, GIF, TIF, and TIFF
- At least two labeled classes are required for training

## Verification

Run the Django system check and test suite:

```bash
python manage.py check
python manage.py test
```

## Technology

Python, Django Templates, HTML5, CSS3, vanilla JavaScript, Pillow, NumPy, Pandas, scikit-learn, PyTorch/Torchvision, and PostgreSQL/Neon.
