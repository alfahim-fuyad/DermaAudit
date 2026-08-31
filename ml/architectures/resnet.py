"""ResNet-50 factory used by the training service."""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model(num_classes, pretrained=False):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
