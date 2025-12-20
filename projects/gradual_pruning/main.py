import copy
import torch
import torchvision
from pathlib import Path
import os
from src.lrp import LRPModel
from src.relevance_accumulator import RelevanceAccumulator
from src.data import get_data_loader
from src.pruning.lrp_gradual_pruner import gradual_lrp_pruning
from src.pruning.finetune import short_finetune
from src.pruning.eval import evaluate_model
from src.models.vgg_cifar import vgg11_cifar100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



base_model = vgg11_cifar100(pretrained=False)
state_dict = torch.load(
    "models/vgg11_cifar100_finetuned.pt",
    map_location=device
)
base_model.load_state_dict(state_dict)
base_model = base_model.to(device)
base_model.eval()

model = copy.deepcopy(base_model).to(device)

class CFG:
    batch_size = 32
    resize = 32

cfg = CFG()

test_loader = get_data_loader(batch_size=cfg.batch_size, resize=cfg.resize, train=False)
train_loader = get_data_loader(batch_size=cfg.batch_size, resize=cfg.resize, train=True)

# مسیر ذخیره relevance scores برای هر step
RELEVANCE_CACHE_DIR = "cache/relevance"
os.makedirs(RELEVANCE_CACHE_DIR, exist_ok=True)  # فولدر cache رو بساز اگر نباشه

print("Data loaders ready:")
print(f"  - Test loader: {len(test_loader.dataset)} samples (for LRP relevance & evaluation)")
print(f"  - Train loader: {len(train_loader.dataset)} samples (for fine-tuning)\n")

for step in range(5):
    print(f"\n{'='*60}")
    print(f"GRADUAL PRUNING STEP {step+1}/5")
    print(f"{'='*60}")
 
    # مسیر cache برای این step
    cache_path = os.path.join(RELEVANCE_CACHE_DIR, f"relevance_step_{step+1}.pt")

    if os.path.exists(cache_path):
        print("[1/4] Loading cached LRP relevance scores...")
        scores = torch.load(cache_path, map_location=device)
        print(f"  Cached relevance loaded from '{cache_path}'\n")
    else:
        print("[1/4] Computing LRP relevance scores on test set...")

        acc = RelevanceAccumulator()
        lrp = LRPModel(model)
        lrp.set_accumulator(acc)

        total_samples = len(test_loader.dataset)
        processed = 0
        batch_count = 0

        for x, _ in test_loader:
            x = x.to(device)
            lrp.forward(x)
            acc.increment(x.size(0))

            processed += x.size(0)
            batch_count += 1

            if batch_count % 20 == 0 or processed == total_samples:
                percentage = (processed / total_samples) * 100
                print(f"  LRP progress: {processed}/{total_samples} samples ({percentage:.1f}%)")

        print(f"  LRP relevance computation completed on {total_samples} samples.\n")

        scores = acc.get_normalized_scores()

        # ذخیره برای دفعات بعدی
        torch.save(scores, cache_path)
        print(f"  Relevance scores saved to cache: '{cache_path}'\n")

    print("[2/4] Performing gradual filter pruning...")
    gradual_lrp_pruning(
        model,
        scores,
        prune_ratio_step=0.1,
        max_total_prune=0.5
    )
    print("  Pruning step completed.\n")

    print("[3/4] Short fine-tuning on train set (1 epoch)...")
    short_finetune(
        model,
        train_loader,
        device,
        epochs=1
    )
    print("  Fine-tuning completed.\n")

    print("[4/4] Evaluating pruned model on test set...")
    evaluate_model(model, test_loader, device)
    print(f"{'='*60}\n")