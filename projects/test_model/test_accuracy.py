import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.models.vgg_cifar import vgg11_cifar100  # دقیقاً این مسیر

# دستگاه
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# مدل رو بساز (pretrained=False چون وزن‌ها رو خودمون لود می‌کنیم)
model = vgg11_cifar100(pretrained=False).to(device)

# لود وزن‌های دانلودشده
model_path = "models/vgg11_cifar100_finetuned.pt"  # مسیر دقیق فایل دانلودشده
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# دیتالودر تست CIFAR-100
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

testset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

# ارزیابی دقت
@torch.no_grad()
def test_accuracy():
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

acc = test_accuracy()
print(f"\nTest Accuracy of loaded model: {acc:.2f}%")

if acc >= 60.0:
    print("🎉 مدل عالیه! می‌تونی بری سراغ pruning و LRP.")
else:
    print("⚠️ دقت پایین‌تر از انتظار هست. چک کن فایل درست رو دانلود کردی یا آموزش کامل بوده.")