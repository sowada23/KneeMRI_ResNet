import torch
import torch.nn as nn
from torchvision import models


def Resnet34(cfg):
    model = models.resnet34(weights=cfg.MODEL_WEIGHT)
    model.fc = nn.Sequential(
          nn.Dropout(p=cfg.DROPOUT),
          nn.Linear(model.fc.in_features, 1)
    )
    return model