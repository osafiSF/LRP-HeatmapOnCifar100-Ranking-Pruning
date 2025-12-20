import torch
import torch.nn.functional as F
import time



def short_finetune(
    model,
    dataloader,
    device,
    epochs=1,
    lr=1e-4
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dataloader)

    for ep in range(epochs):
        print(f"  Starting fine-tune epoch {ep+1}/{epochs}...")
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader, 1):  # از 1 شروع کن
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 🔥 گزارش زود به زود: هر ۵۰ بچ یا آخرین بچ
            if batch_idx % 50 == 0 or batch_idx == total_batches:
                avg_loss_so_far = total_loss / batch_idx
                elapsed = time.time() - start_time
                eta = (elapsed / batch_idx) * (total_batches - batch_idx) if batch_idx < total_batches else 0
                print(
                    f"    Batch {batch_idx}/{total_batches} | "
                    f"Avg loss: {avg_loss_so_far:.4f} | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {eta:.1f}s"
                )

        final_avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        print(f"  Fine-tune epoch {ep+1}/{epochs} completed | Final avg loss: {final_avg_loss:.4f} | Time: {epoch_time:.1f}s\n")

    model.eval()