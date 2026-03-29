from torchvision import transforms
import torch

# def repeat_to_3ch(x):
#     return x.repeat(3, 1, 1)

class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

def build_train_transforms(cfg):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.Lambda(lambda x: torch.clamp(x, -3.0, 3.0)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
        AddGaussianNoise(std=0.01)
        
    ])


def build_eval_transforms(cfg):
    return transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.Lambda(lambda x: torch.clamp(x, -3.0, 3.0)),
         transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])