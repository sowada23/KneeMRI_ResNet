import torch
import torch.nn as nn
from torchvision import models


def Resnet50(cfg):
    model = models.resnet50(weights=cfg.MODEL_WEIGHT)
    old_conv = model.conv1  # original: [64, 3, 7, 7]
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 1)
    )
    return model