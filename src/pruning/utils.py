import torch
import torch.nn as nn

import torch
import torch.nn as nn

def prune_conv_layer(model, layer_name, prune_indices):
    """
    Prunes filters from a Conv2d layer and fixes the following BatchNorm2d and next Conv2d if present.
    """
    modules = dict(model.named_modules())
    conv = modules[layer_name]

    keep_indices = [i for i in range(conv.out_channels) if i not in prune_indices]
    num_keep = len(keep_indices)

    if num_keep == 0:
        raise ValueError(f"All filters pruned in layer {layer_name}! Cannot proceed.")

    print(f"    Pruning '{layer_name}': keeping {num_keep}/{conv.out_channels} filters")

    # Prune Conv2d
    conv.weight = nn.Parameter(conv.weight.data[keep_indices])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_indices])
    conv.out_channels = num_keep

    # Find and fix the next layers (BatchNorm then next Conv)
    found_conv = False
    bn_fixed = False
    next_conv_fixed = False

    for name, m in model.named_modules():
        if found_conv:
            if isinstance(m, nn.BatchNorm2d) and not bn_fixed:
                # Prune BatchNorm
                m.weight = nn.Parameter(m.weight.data[keep_indices])
                m.bias = nn.Parameter(m.bias.data[keep_indices])
                m.running_mean = m.running_mean[keep_indices]
                m.running_var = m.running_var[keep_indices]
                m.num_features = num_keep
                bn_fixed = True
                print(f"    Fixed BatchNorm after '{layer_name}' → {num_keep} features")

            elif isinstance(m, nn.Conv2d) and not next_conv_fixed:
                # Fix next Conv in_channels
                m.weight = nn.Parameter(m.weight.data[:, keep_indices, :, :])
                m.in_channels = num_keep
                next_conv_fixed = True
                print(f"    Fixed next Conv '{name}' in_channels → {num_keep}")
                break  # معمولاً بعد از Conv بعدی دیگه نیاز نیست ادامه بدیم

        if name == layer_name:
            found_conv = True