import torch.nn as nn

def get_vgg_conv_layers(model):
    """
    Returns ordered list of (layer_id, layer_name, conv_layer)
    """
    conv_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layers.append((name, layer))
    return conv_layers
