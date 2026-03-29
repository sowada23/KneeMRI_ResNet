import torch
import torch.nn as nn
from torchvision import models


def Resnet50(cfg):
    model = models.resnet50(weights=cfg.MODEL_WEIGHT)
    
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
          in_channels=1,
          out_channels=old_conv.out_channels,
          kernel_size=old_conv.kernel_size,
          stride=old_conv.stride,
          padding=old_conv.padding,
          bias=False,
    )
    
    with torch.no_grad():
          model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    
    model.fc = nn.Sequential(
          nn.Dropout(p=0.3),
          nn.Linear(model.fc.in_features, 1)
    )
    return model