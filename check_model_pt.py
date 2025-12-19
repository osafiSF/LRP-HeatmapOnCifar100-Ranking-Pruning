import torch

MODEL_PATH = "models/vgg11.pt"

# Load full model (weights_only=False)
obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

print("Loaded object type:")
print(type(obj))

# اگر دیکشنری بود، چند تا کلیدش رو چاپ کن
if isinstance(obj, dict):
    print("\nThis looks like a state_dict. Sample keys:")
    for i, k in enumerate(obj.keys()):
        print(k)
        if i == 5:
            break
else:
    print("\nThis looks like a full model.")
    print("Has features:", hasattr(obj, "features"))
    print("Has classifier:", hasattr(obj, "classifier"))
