from src.lrp import LRPModel
from src.relevance_accumulator import RelevanceAccumulator
from src.data import get_data_loader
from src.ranking.vgg_mapping import get_vgg_conv_layers
from src.ranking.filter_ranking import rank_filters

import argparse
import torch
import torchvision
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model_path = Path(__file__).resolve().parents[2] / "models" / "VGG11.pt"

with torch.serialization.safe_globals([torchvision.models.vgg.VGG]):
    model = torch.load(model_path, map_location=device, weights_only=False)

model.to(device)
model.eval()


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--resize", default=32, type=int)
config = parser.parse_args([])  # چون فعلاً CLI نداریم

dataloader = get_data_loader(config)


acc = RelevanceAccumulator()
lrp_model = LRPModel(model, top_k=0.0)
lrp_model.set_accumulator(acc)

for i, (x, y) in enumerate(dataloader):
    x = x.to(device)
    lrp_model.forward(x)
    acc.increment(x.size(0))
    if i % 10 == 0:
        print(
            f"[{i:04d}] "
            f"Processed samples: {acc.count}"
        )

scores = acc.get_normalized_scores()
conv_layers = get_vgg_conv_layers(model)

rankings = rank_filters(scores, conv_layers)

print("\n=== Filter-wise Ranking Summary ===")
for layer_name, info in rankings.items():
    print(
        f"{layer_name}: "
        f"filters={info['num_filters']}, "
        f"top_score={info['sorted_scores'][0]:.6f}, "
        f"least_score={info['sorted_scores'][-1]:.6f}"
    )
print("Number of layers with accumulated relevance:", len(scores))

# ==== چاپ جزئیات همه فیلترها ====
print("\n=== Filter-wise Detailed Scores ===")
for layer_name, info in rankings.items():
    print(
        f"\nLayer: {layer_name} (num_filters={info['num_filters']})" 
        f"top_score={info['sorted_scores'][0]:.4f}, "
        f"least_score={info['sorted_scores'][-1]:.4f}"
        )
    # for i, score in enumerate(info['sorted_scores']):
    #     print(f"  Filter {info['sorted_indices'][i]}: score={score:.6f}")

# print("\n=== Filter-wise Ranking ===")
# for layer_name, info in rankings.items():
#     print(
#         f"{layer_name}: "
#         f"filters={info['num_filters']}, "
#         f"top_score={info['sorted_scores'][0]:.4f}, "
#         f"least_score={info['sorted_scores'][-1]:.4f}"
#     )
# print("Number of layers with accumulated relevance:", len(scores))
# for layer_id, score in scores.items():
#     print(score.shape, score.mean().item(), score.max().item())