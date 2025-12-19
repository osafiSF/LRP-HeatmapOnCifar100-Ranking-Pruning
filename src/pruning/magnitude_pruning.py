import torch
import torch.nn as nn
from .lrp_pruning import prune_conv_layer, _fix_vgg_classifier

def magnitude_pruning(model, prune_ratio=0.2):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):

            weight_norm = layer.weight.data.abs().sum(dim=(1, 2, 3))
            num_filters = weight_norm.numel()
            k = int(prune_ratio * num_filters)

            prune_indices = torch.argsort(weight_norm)[:k].tolist()

            print(f"[MAG-PRUNE] {name}: pruning {k}/{num_filters}")

            prune_conv_layer(model, name, prune_indices)

    # fix classifier like LRP
    _fix_vgg_classifier(model)

    return model
