# import torch
# import torch.nn as nn
# import torch.optim as optim
# from src.models.vgg_cifar import vgg11_cifar100
# from src.data import get_data_loader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = vgg11_cifar100(pretrained=True).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(
#     model.parameters(),
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4
# )

# train_loader = get_data_loader(train=True, batch_size=128)
# test_loader  = get_data_loader(train=False, batch_size=128)

# for epoch in range(30):
#     model.train()
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         loss = criterion(model(x), y)
#         loss.backward()
#         optimizer.step()

#     print(f"[Epoch {epoch}] done")

# torch.save(model.state_dict(), "models/vgg11_cifar100_finetuned.pt")

# if __name__ == '__main__':
#     main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from src.models.vgg_cifar import vgg11_cifar100
from src.data import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # مدل
    model = vgg11_cifar100(pretrained=True).to(device)

    # loss و optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,           # شروع با lr=0.01 خوبه برای fine-tune از ImageNet
        momentum=0.9,
        weight_decay=5e-4
    )

    # 🔥 Scheduler اضافه شده اینجا
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # معنی: هر 50 epoch، lr رو ضربدر 0.1 کن (یعنی 0.01 → 0.001 → 0.0001)

    # config برای batch_size
    class Config:
        batch_size = 128
        resize = None  # یا 32 اگر می‌خوای resize کنی

    config = Config()

    train_loader = get_data_loader(batch_size=128, resize=None, train=True)
    test_loader = get_data_loader(batch_size=128, resize=None, train=False)

    num_epochs = 150  # توصیه: حداقل 100-150 epoch برای دقت خوب

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # چاپ loss
        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1:03d}/{num_epochs}] Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 🔥 scheduler.step() دقیقاً اینجا، بعد از هر epoch
        scheduler.step()

        # اختیاری: هر 20 epoch دقت روی test چک کن
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            acc = 100 * correct / total
            print(f"    >>> Test Accuracy: {acc:.2f}%\n")

    # ذخیره مدل
    save_path = "models/vgg11_cifar100_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining finished! Model saved to {save_path}")

if __name__ == '__main__':
    main()