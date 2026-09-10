# Methodology

The intended lifecycle is validate → profile → harmonize → audit → leakage control → bias/imbalance review → compare models → evaluate reliability → explain → register. Dataset records preserve the profile and audit JSON returned by each stage so reports can remain tied to evidence.

All prediction results recommend qualified human review. A future production checkpoint runner should load one of the Torchvision factories, apply calibrated probabilities, and write Grad-CAM artifacts before activating a model in the registry.
