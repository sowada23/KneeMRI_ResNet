from pathlib import Path


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def print_trainable_params(model, save_path=None):
    lines = []
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            lines.append(f"{name:50s} | shape={tuple(param.shape)} | params={n}")

    lines.append("-" * 80)
    lines.append(f"Trainable params: {trainable:,}")
    lines.append(f"Total params:     {total:,}")
    lines.append(f"Percent trainable: {100 * trainable / total:.2f}%")

    text = "\n".join(lines)

    print(text)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text)

    return text

def setup_fc_only(model):
    freeze_all(model)
    unfreeze_module(model.fc)
    return model

def setup_layer4_fc(model):
    freeze_all(model)
    unfreeze_module(model.layer4[1]) 
    unfreeze_module(model.layer4[2]) 
    unfreeze_module(model.fc)
    return model

def setup_layer4_layer3_fc(model):
    freeze_all(model)
    unfreeze_module(model.layer3[0])
    unfreeze_module(model.layer3[1])
    unfreeze_module(model.layer3[2])
    unfreeze_module(model.layer3[3])
    unfreeze_module(model.layer3[4])
    unfreeze_module(model.layer3[5])
    unfreeze_module(model.layer4)
    unfreeze_module(model.fc)
    return model

def setup_layer4_layer3_layer2_fc(model):
    freeze_all(model)
    # unfreeze_module(model.layer2[0])
    unfreeze_module(model.layer2[1])
    unfreeze_module(model.layer2[2])
    unfreeze_module(model.layer2[3])
    unfreeze_module(model.layer3)
    unfreeze_module(model.layer4)
    unfreeze_module(model.fc)
    return model