from torchvision import transforms
import torch

class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

def repeat_to_3ch(x):
    return x.repeat(3, 1, 1)

def build_train_transforms(cfg):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.Lambda(lambda x: torch.clamp(x, -3.0, 3.0)),
        transforms.Lambda(repeat_to_3ch),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
        AddGaussianNoise(std=0.01),
    ])


def build_eval_transforms(cfg):
    return transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.Lambda(lambda x: torch.clamp(x, -3.0, 3.0)),
         transforms.Lambda(repeat_to_3ch),
         transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])