"""Class for layer-wise relevance propagation.

Layer-wise relevance propagation for VGG-like networks from PyTorch's Model Zoo.
Implementation can be adapted to work with other architectures as well by adding the corresponding operations.

    Typical usage example:

        model = torchvision.models.vgg16(pretrained=True)
        lrp_model = LRPModel(model)
        r = lrp_model.forward(x)

"""
from copy import deepcopy

import torch
from torch import nn

from src.utils import layers_lookup


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module, top_k: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network
        self.layers = self._get_layer_operations()
        self.accumulator = None

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()
        
    def set_accumulator(self, accumulator):
        self.accumulator = accumulator
        for layer in self.lrp_layers:
            if hasattr(layer, "set_accumulator"):
                layer.set_accumulator(accumulator)


    def _create_lrp_model(self) -> torch.nn.ModuleList:
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()

        lrp_layers = torch.nn.ModuleList()

        for name, layer in layers[::-1]:
            try:
                lrp_layer = lookup_table[type(layer)](
                    layer=layer,
                    top_k=self.top_k,
                    layer_name=name  # پاس دادن name به لایه LRP
                )
                lrp_layers.append(lrp_layer)
            except KeyError:
                raise NotImplementedError(f"LRP not implemented for {type(layer)}")


        return lrp_layers

    # def _get_layer_operations(self) -> torch.nn.ModuleList:
    #     """Get all network operations and store them in a list.

    #     This method is adapted to VGG networks from PyTorch's Model Zoo.
    #     Modify this method to work also for other networks.

    #     Returns:
    #         Layers of original model stored in module list.

    #     """
    #     layers = torch.nn.ModuleList()

    #     # Parse VGG-16
    #     for layer in self.model.features:
    #         layers.append(layer)

    #     layers.append(self.model.avgpool)
    #     layers.append(torch.nn.Flatten(start_dim=1))

    #     for layer in self.model.classifier:
    #         layers.append(layer)

    #     return layers

    def _get_layer_operations(self) -> list:
        layers = []

        # feature layers
        for name, layer in self.model.features._modules.items():
            layers.append((f"features.{name}", layer))

        # avgpool
        layers.append(("avgpool", self.model.avgpool))

        # flatten
        layers.append(("flatten", torch.nn.Flatten(start_dim=1)))

        # classifier layers
        for name, layer in self.model.classifier._modules.items():
            layers.append((f"classifier.{name}", layer))

        return layers


    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).

        """
        activations = []

        with torch.no_grad():
            activations.append(torch.ones_like(x))
            for name, layer in self.layers:
                x = layer(x)
                activations.append(x)

        # reverse activations (output first)
        activations = activations[::-1]
        activations = [a.requires_grad_(True) for a in activations]

        relevance = torch.softmax(activations.pop(0), dim=-1)

        assert len(self.lrp_layers) == len(activations), (
            f"LRP layers ({len(self.lrp_layers)}) "
            f"!= activations ({len(activations)})"
        )

        for layer, activation in zip(self.lrp_layers, activations):
            relevance = layer(activation, relevance)

        #return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()
        return relevance
