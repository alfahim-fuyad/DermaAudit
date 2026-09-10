# DermaAudit AI

DermaAudit AI is a server-rendered Django workspace for **generic image-classification dataset auditing and training**. Skin-lesion research is the flagship use case, but the same evidence-first workflow can be used for plant disease, X-ray, animal, food, or other labeled image datasets.

The platform is designed around one final, fixed flow — **Part 1** is a generic dataset audit pipeline that works on any labeled image dataset, and **Part 2** is a separate prediction module for the flagship HAM10000 skin-lesion use case:

```text
PART 1 · GENERIC DATASET AUDIT PIPELINE (any image classification dataset)

  1. DATASET INGESTION            Upload a ZIP, read files and folders
  2. DATASET AUTO-DETECTION       Detect structure, classes, image-label mapping
  3. DATA AUDIT                   File/image quality · label/data quality ·
                                  distribution quality
  4. AUDIT REPORT                 Issues found, issue count, severity,
                                  recommended action
  5. AUTOMATIC CLEANING           Remove corrupted, duplicate, and invalid files;
                                  handle missing labels and unknown files
  6. DATASET VALIDATION           Re-check images, labels, class counts, and
                                  readability on the cleaned set
  7. PREPROCESSING                Path mapping, load, resize, RGB conversion,
                                  pixel normalization, label encoding
  8. DATASET SPLITTING            Train / Validation / Test
  9. TRAINING PREPARATION         Class-imbalance handling, augmentation
                                  (training set only), one-hot encoding
 10. TRAINING-READY DATASET       Train + Validation + Test partitions

PART 2 · SEPARATE PREDICTION MODULE (HAM10000 flagship use case)

  HAM10000 dataset → local training → saved model → website
  → upload skin image → preprocessing → HAM10000 model
  → disease prediction → predicted class + confidence
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
- Multiple image-part folders paired with metadata, such as HAM10000:

  ```text
  HAM10000/
  ├── HAM10000_images_part_1/
  ├── HAM10000_images_part_2/
  ├── HAM10000_metadata.csv
  └── hmnist_28_28_L.csv
  ```

  The scanner recursively finds images in both parts, maps `image_id` to the image filename, and uses `dx` as the class label. Extra tabular exports such as `hmnist_*.csv` are detected and marked unused when the original JPEGs are available.

Before training, the validator checks supported formats, archive safety, readable images, corrupted files, dimensions, color modes, missing labels, class count, duplicate filenames, exact duplicate content, and near-duplicate content. The original archive is never modified, and image-part folders are not mistaken for class labels when metadata is present.

### 2. Profile, harmonize, and audit

The dataset profile records image formats, resolutions, channels, class distribution, metadata files and columns, available IDs, group information, and subgroup fields. Labels are normalized without replacing the original upload.

The audit combines:

- Image quality and validation findings
- Exact and perceptual duplicate signals
- Label and metadata consistency
- Class imbalance and weighted-training recommendations
- Group/entity identifiers for leakage control
- Bias and fairness readiness when approved subgroup metadata exists

The audit report lists every issue with its **count**, **severity** (High / Medium / Low), and a **recommended action**, grouped into file/image quality, label/data quality, and distribution quality.

Datasets with blocking validation errors remain **Needs review** and cannot be used for training.

### 3. Automatic cleaning and re-check

After the audit, the pipeline automatically cleans the audited records into a training-ready manifest:

- Corrupted and unreadable images are excluded (they never enter the record set)
- Unknown or unsupported files are ignored
- Exact duplicates are removed, keeping the first copy
- Near-duplicates are removed before splitting
- Low-resolution images are removed
- Images without a class label are excluded

The cleaned manifest is then re-checked: image readability, class labels, class count (at least two), class coverage, and rare-class warnings. A dataset is only **Ready** when validation passes **and** the cleaned-set re-check passes. As a safety guard, an archive where more than 25% of the images are unreadable stays **Needs review**. The original archive is never modified — cleaning produces the manifest used for splitting and training.

### 4. Preprocess, split, and prepare training

The default split is **70% train / 15% validation / 15% test**. Training enforces that the values total 100%.

The default split is **70% train / 15% validation / 15% test**. Training enforces that the values total 100%.

- When matched patient, lesion, subject, case, study, or entity IDs exist, records are kept together with a group-aware split.
- Without group IDs, the pipeline uses a deterministic class-aware stratified split.
- Preprocessing maps image paths, loads files, resizes to the model input size, converts to RGB, and normalizes pixels; labels are encoded to class indices.
- Augmentation is applied only to training inputs; validation and test inputs remain untouched.
- Class imbalance is handled with class-weighted loss, and one-hot encoding is used for calibration metrics.

Each experiment can compare a declared intervention. **Cleaned dataset (recommended)** is the default and matches the automatic cleaning rules from the audit pipeline:

| Intervention | Training candidate |
| --- | --- |
| Original | Use the validated upload as-is (control) |
| Duplicate-free | Exclude exact duplicate images |
| Leakage-controlled | Use group-aware splitting when group IDs exist |
| Bias-mitigated | Preserve subgroup evidence for review |
| Quality-controlled | Exclude low-resolution images |
| Imbalance-handled | Use class-weighted loss |
| Cleaned dataset (recommended) | The automatically cleaned manifest: duplicates, near-duplicates, low-quality, and unlabelled samples removed |

### 5. Train and evaluate

Choose one of the supported architecture profiles—EfficientNet-B0, ResNet-50, or MobileNetV3—and start an experiment from **Model training**. The CPU training runner records:

- Train loss and train Macro-F1 per epoch
- Validation Macro-F1 and learning rate per epoch
- AdamW optimization and learning-rate scheduling
- Early stopping and the best validation epoch
- Held-out accuracy, Macro-F1, balanced accuracy, precision, recall, AUROC, confusion matrix, and per-class metrics

The complete run record is available from **Reports → Model report**. Model selection should consider Macro-F1 and minority-class behavior, not accuracy alone.

### 6. Reliability, registry, and prediction (Part 2)

Prediction is a **separate module** from the generic audit pipeline. The flagship path trains locally on the HAM10000 skin-lesion dataset through Part 1, registers the saved checkpoint, and then serves skin-image predictions from the website:

```text
HAM10000 dataset → local training → saved model → website
→ upload skin image → preprocessing → HAM10000 model
→ disease prediction → predicted class + confidence
```

Each completed run saves a versioned checkpoint with its class mapping, preprocessing settings, split plan, intervention, evaluation metrics, calibration evidence, and dataset lineage. Select one checkpoint from **Experiments** before requesting a prediction.

The prediction review:

- Validates the uploaded image before inference
- Applies calibrated confidence when calibration data exists
- Shows the predicted class and top alternatives
- Abstains when confidence or top-class margin is too low
- Generates a Grad-CAM overlay when the checkpoint supports it
- Always recommends qualified human review

### 7. Monitoring

**Reports** provides live workspace signals for dataset readiness, training activity, prediction volume, abstentions, average confidence, and review queue size. Data drift is compared with the active model’s training class distribution only when both inputs exist.

Fairness drift and performance-after-deployment analysis remain explicitly unavailable until approved subgroup metadata and verified labels are available. Predictions are not treated as ground truth.

## Upload limits and supported images

- ZIP archives up to 7 GB; larger datasets may require local processing
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
