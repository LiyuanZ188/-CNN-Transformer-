import torchvision.models as models
from transformers import (
    ViTForImageClassification,
    SwinForImageClassification,
)
import torch.nn as nn

def get_model(model_name, num_classes):
    if model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    elif model_name == "swin":
        model = SwinForImageClassification.from_pretrained(
            "microsoft/swin-base-patch4-window7-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

