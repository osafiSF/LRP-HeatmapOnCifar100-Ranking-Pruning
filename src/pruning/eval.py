import torch

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_batches = len(dataloader)

    print("  Starting evaluation on test set...")

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # گزارش هر ۵۰ بچ
        if batch_idx % 50 == 0 or batch_idx == total_batches:
            current_acc = (correct / total) * 100 if total > 0 else 0
            print(f"    Evaluated {batch_idx}/{total_batches} batches | Current accuracy: {current_acc:.2f}%")

    final_acc = correct / total
    print(f"  >>> Final Test Accuracy: {final_acc:.4f} ({correct}/{total} correct)\n")
    return final_acc