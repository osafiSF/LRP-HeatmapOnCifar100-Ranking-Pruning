import torch

def rank_filters(relevance_scores: dict, conv_layers: list):
    """
    relevance_scores: {layer_id: Tensor[C]}
    conv_layers: list of (layer_id, layer_name, conv_layer)

    Returns:
        ranking dict per layer
    """
    rankings = {}

    for layer_name, layer in conv_layers:
        if layer_name not in relevance_scores:
            continue

        scores = relevance_scores[layer_name]
        values, indices = torch.sort(scores, descending=True)

        rankings[layer_name] = {
            "layer_name": layer_name,
            "num_filters": scores.numel(),
            "sorted_indices": indices,
            "sorted_scores": values,
        }

    return rankings
