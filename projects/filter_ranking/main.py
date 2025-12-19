from src.lrp import LRPModel
from src.relevance_accumulator import RelevanceAccumulator
from src.data import get_data_loader
from src.ranking.vgg_mapping import get_vgg_conv_layers
from src.ranking.filter_ranking import rank_filters
from src.eval.metrics import evaluate_accuracy, measure_speed
from src.pruning.lrp_pruning import lrp_based_pruning
from src.pruning.magnitude_pruning import magnitude_pruning
from pathlib import Path
import argparse
import torch
import torchvision
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model_path = Path(__file__).resolve().parents[2] / "models" / "VGG11.pt"

with torch.serialization.safe_globals([torchvision.models.vgg.VGG]):
    model = torch.load(model_path, map_location=device, weights_only=False)

model.to(device)
model.eval()

original_model = model
model_lrp = copy.deepcopy(original_model)
model_mag = copy.deepcopy(original_model)


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

print("\n===== BASE MODEL EVALUATION =====")
base_acc = evaluate_accuracy(original_model, dataloader, device)
base_speed = measure_speed(original_model, dataloader, device)

print(f"Accuracy: {base_acc:.4f}")
print(f"Speed: {base_speed}")

print("\n===== LRP PRUNING =====")
model_lrp = lrp_based_pruning(model_lrp, rankings, prune_ratio=0.2)

lrp_acc = evaluate_accuracy(model_lrp, dataloader, device)
lrp_speed = measure_speed(model_lrp, dataloader, device)

print(f"Accuracy: {lrp_acc:.4f}")
print(f"Speed: {lrp_speed}")

print("\n===== MAGNITUDE PRUNING =====")
model_mag = magnitude_pruning(model_mag, prune_ratio=0.2)

mag_acc = evaluate_accuracy(model_mag, dataloader, device)
mag_speed = measure_speed(model_mag, dataloader, device)

print(f"Accuracy: {mag_acc:.4f}")
print(f"Speed: {mag_speed}")

print("\n===== FINAL COMPARISON =====")
print(f"Base      | Acc={base_acc:.4f} | Thr={base_speed['throughput_img_per_sec']:.1f}")
print(f"LRP-Prune| Acc={lrp_acc:.4f} | Thr={lrp_speed['throughput_img_per_sec']:.1f}")
print(f"Mag-Prune| Acc={mag_acc:.4f} | Thr={mag_speed['throughput_img_per_sec']:.1f}")

# # ==== چاپ جزئیات همه فیلترها ====
# print("\n=== Filter-wise Detailed Scores ===")
# for layer_name, info in rankings.items():
#     print(
#         f"\nLayer: {layer_name} (num_filters={info['num_filters']})" 
#         f"top_score={info['sorted_scores'][0]:.4f}, "
#         f"least_score={info['sorted_scores'][-1]:.4f}"
#         )
    


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