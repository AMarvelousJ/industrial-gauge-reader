from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


def build_model(class_count: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def build_transforms(image_size: int = 224):
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(8),
            v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    evaluation = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return train, evaluation


def load_checkpoint(path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    styles = checkpoint["styles"]
    model = build_model(len(styles), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, styles, checkpoint

