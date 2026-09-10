# Methodology

The intended lifecycle is fixed: ingest → auto-detect → audit → audit report → automatic cleaning → cleaned-dataset re-check → preprocess → split → training preparation → training-ready dataset → compare models → evaluate reliability → explain → register. Dataset records preserve the profile, audit, and cleaning JSON returned by each stage so reports can remain tied to evidence. Prediction (HAM10000 flagship) is a separate module that consumes registered checkpoints.

All prediction results recommend qualified human review. A future production checkpoint runner should load one of the Torchvision factories, apply calibrated probabilities, and write Grad-CAM artifacts before activating a model in the registry.
