
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained checkpoint
checkpoint = torch.load(
    "models/mobilenetv3_food_v3.pth",
    map_location=device,
    weights_only=True
)
classes = checkpoint["classes"]

# Rebuild the same model architecture used during training
model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Preprocess the image
transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
image = Image.open("bigMac.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    top_probs, top_indices = torch.topk(probabilities, 3, dim=1)

print("Top 3 predictions:")
for probability, index in zip(top_probs[0], top_indices[0]):
    print(f"{classes[index.item()]}: {probability.item() * 100:.2f}%")
