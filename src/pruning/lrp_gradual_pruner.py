import torch
from src.ranking.filter_ranking import rank_filters
from src.ranking.vgg_mapping import get_vgg_conv_layers
from src.pruning.utils import prune_conv_layer

# def gradual_lrp_pruning(
#     model,
#     relevance_scores,
#     prune_ratio_step=0.1,
#     max_total_prune=0.5
# ):
#     conv_layers = get_vgg_conv_layers(model)
#     rankings = rank_filters(relevance_scores, conv_layers)

#     current_pruned = 0.0
#     layer_dict = {name: layer for _, name, layer in conv_layers}

#     for layer_name, info in rankings.items():
#         if current_pruned >= max_total_prune:
#             break

#         layer = layer_dict[layer_name]
#         num_filters = info["num_filters"]
#         k = int(prune_ratio_step * num_filters)

#         prune_idx = info["sorted_indices"][-k:]

#         print(
#             f"[GRADUAL-PRUNE] {layer_name}: "
#             f"step_prune={k}/{num_filters}"
#         )

#         prune_conv_layer(
#             model,
#             layer_name,
#             prune_idx
#         )

#         current_pruned += prune_ratio_step
def gradual_lrp_pruning(
    model,
    relevance_scores,
    prune_ratio_step=0.1,
    max_total_prune=0.5
):
    conv_layers = get_vgg_conv_layers(model)
    rankings = rank_filters(relevance_scores, conv_layers)

    current_pruned_ratio = 0.0
    layer_dict = {name: layer for name, layer in conv_layers}
    

    print(f"  Starting pruning (step ratio: {prune_ratio_step}, max total: {max_total_prune})")

    for layer_name, info in rankings.items():
        if current_pruned_ratio >= max_total_prune:
            print(f"  Reached max prune ratio ({max_total_prune}). Stopping.")
            break

        num_filters = info["num_filters"]
        k = int(prune_ratio_step * num_filters)

        if k == 0:
            continue

        print(
            f"  Pruning layer '{layer_name}': "
            f"{k} filters out of {num_filters} ({prune_ratio_step*100:.1f}% of layer)"
        )

        prune_idx = info["sorted_indices"][-k:].tolist()

        prune_conv_layer(model, layer_name, prune_idx)

        current_pruned_ratio += prune_ratio_step

    print(f"  Current total pruned ratio: {min(current_pruned_ratio, max_total_prune):.2f}\n")