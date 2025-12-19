import time
import torch

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


@torch.no_grad()
def measure_speed(model, dataloader, device, num_batches=50):
    model.eval()

    # warm-up
    for i, (x, _) in enumerate(dataloader):
        if i >= 5:
            break
        model(x.to(device))

    start = time.time()
    total_images = 0

    for i, (x, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        x = x.to(device)
        model(x)
        total_images += x.size(0)

    elapsed = time.time() - start
    latency = elapsed / num_batches
    throughput = total_images / elapsed

    return {
        "latency_sec_per_batch": latency,
        "throughput_img_per_sec": throughput,
    }
