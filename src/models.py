import torch.nn as nn
from torchvision import models

from .dataset import NUM_CLASSES


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),   # 224 -> 112
            block(32, 64),            # 112 -> 56
            block(64, 128),           # 56  -> 28
            block(128, 256),          # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # -> (N, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _set_backbone_trainable(module, trainable):
    for p in module.parameters():
        p.requires_grad = trainable


def build_resnet18(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    if freeze_backbone:
        _set_backbone_trainable(net, False)  # frozen feature extractor
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def build_model(name, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained, freeze_backbone)
    raise ValueError(f"unknown model name: {name!r}")


def gradcam_target_layer(model, name):
    """Return the conv layer Grad-CAM should hook for a given model.
    """
    name = name.lower()
    if name == "simple_cnn":
        return model.features[-1]        # last conv block
    if name == "resnet18":
        return model.layer4[-1]          # last residual block
    raise ValueError(f"unknown model name: {name!r}")
