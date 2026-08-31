# Architecture

DermaAudit AI is a single Django project. Views persist small, auditable records in the database and render HTML templates directly. Dataset uploads are stored under `MEDIA_ROOT`; the production database can be Neon PostgreSQL through `DATABASE_URL`, while local preview defaults to SQLite.

The ML boundary is intentionally explicit. `ml/architectures/` contains Torchvision transfer-learning factories, `ml/preprocessing/` owns input transforms, and `ml/evaluation/` owns scikit-learn metrics. The prediction screen labels its no-checkpoint path as preview analysis rather than presenting a heuristic as a medical model.
