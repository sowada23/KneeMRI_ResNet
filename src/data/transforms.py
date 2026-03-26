from torchvision import transforms
import torch

class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

def build_train_transforms(cfg):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        AddGaussianNoise(std=0.02),
    ])


def build_eval_transforms(cfg):
    return transforms.Compose([
         transforms.Resize((224, 224)),
    ])