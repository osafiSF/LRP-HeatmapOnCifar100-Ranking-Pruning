import torch
import torch.nn as nn

# def prune_conv_layer(model, layer_name, prune_indices):
#     """
#     Removes filters (out_channels) from a Conv2d layer
#     and fixes the next Conv2d layer's in_channels.
#     """
#     modules = dict(model.named_modules())
#     conv = modules[layer_name]

#     keep_indices = [i for i in range(conv.out_channels) if i not in prune_indices]

#     # prune conv weights
#     conv.weight = nn.Parameter(conv.weight.data[keep_indices, :, :, :])
#     if conv.bias is not None:
#         conv.bias = nn.Parameter(conv.bias.data[keep_indices])

#     conv.out_channels = len(keep_indices)

#     # fix next conv layer
#     found = False
#     for name, m in model.named_modules():
#         if found and isinstance(m, nn.Conv2d):
#             m.weight = nn.Parameter(m.weight.data[:, keep_indices, :, :])
#             m.in_channels = len(keep_indices)
#             break
#         if name == layer_name:
#             found = True

def prune_conv_layer(model, layer_name, prune_indices):
    """
    Prunes a Conv2d layer AND its following BatchNorm2d layer.
    Also fixes the next Conv2d layer's in_channels.
    """
    modules = dict(model.named_modules())
    conv = modules[layer_name]

    keep_indices = [i for i in range(conv.out_channels) if i not in prune_indices]

    # ---- prune Conv2d ----
    conv.weight = nn.Parameter(conv.weight.data[keep_indices])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_indices])
    conv.out_channels = len(keep_indices)

    found = False
    bn_fixed = False

    for name, m in model.named_modules():
        if name == layer_name:
            found = True
            continue

        # ---- prune following BatchNorm ----
        if found and isinstance(m, nn.BatchNorm2d) and not bn_fixed:
            m.weight = nn.Parameter(m.weight.data[keep_indices])
            m.bias = nn.Parameter(m.bias.data[keep_indices])
            m.running_mean = m.running_mean.data[keep_indices]
            m.running_var = m.running_var.data[keep_indices]
            m.num_features = len(keep_indices)
            bn_fixed = True
            continue

        # ---- fix next Conv in_channels ----
        if found and isinstance(m, nn.Conv2d):
            m.weight = nn.Parameter(m.weight.data[:, keep_indices])
            m.in_channels = len(keep_indices)
            break


def _fix_vgg_classifier(model):
    """
    Fixes the first Linear layer of VGG classifier
    after structured Conv pruning.
    """
    # last conv layer output channels
    last_conv = None
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            last_conv = m

    assert last_conv is not None

    out_channels = last_conv.out_channels

    # VGG spatial size after avgpool is 7x7
    new_in_features = out_channels * 7 * 7

    old_linear = model.classifier[0]

    if old_linear.in_features != new_in_features:
        print(
            f"[FIX-CLASSIFIER] Linear in_features: "
            f"{old_linear.in_features} → {new_in_features}"
        )

        model.classifier[0] = nn.Linear(
            new_in_features,
            old_linear.out_features,
        )



def lrp_based_pruning(model, rankings, prune_ratio=0.2):
    """
    rankings: output of rank_filters
    """
    for layer_name, info in rankings.items():
        num_filters = info["num_filters"]
        k = int(prune_ratio * num_filters)

        prune_indices = info["sorted_indices"][-k:].tolist()

        print(f"[LRP-PRUNE] {layer_name}: pruning {k}/{num_filters}")
        prune_conv_layer(model, layer_name, prune_indices)

    # ==== FIX CLASSIFIER INPUT DIM ====
    _fix_vgg_classifier(model)

    return model
