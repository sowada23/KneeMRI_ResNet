def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def print_trainable_params(model):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            print(f"TRAINABLE: {name}")
    print(f"trainable params: {trainable:,} / {total:,}")

def setup_fc_only(model):
    freeze_all(model)
    unfreeze_module(model.fc)
    return model

def setup_layer4_fc(model):
    freeze_all(model)
    unfreeze_module(model.layer4)
    unfreeze_module(model.fc)
    return model

def setup_layer4_layer3_fc(model):
    freeze_all(model)
    unfreeze_module(model.layer3)
    unfreeze_module(model.layer4)
    unfreeze_module(model.fc)
    return model