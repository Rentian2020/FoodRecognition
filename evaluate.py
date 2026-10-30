
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Food101
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
model_path = "models/mobilenetv3_food_v3.pth"

print("Using device:", device)

# Load fine-tuned model checkpoint
checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

selected_classes = checkpoint["classes"]

model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(selected_classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Load Food-101 test dataset
transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

test_data = Food101(
    root="./data",
    split="test",
    transform=transform,
    download=False
)

class_to_new_idx = {
    test_data.class_to_idx[name]: idx
    for idx, name in enumerate(selected_classes)
}


class FoodSubset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [
            i for i, label in enumerate(dataset._labels)
            if label in class_to_new_idx
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, old_label = self.dataset[self.indices[index]]
        return image, class_to_new_idx[old_label]


test_subset = FoodSubset(test_data)

test_loader = DataLoader(
    test_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_subset))

# Evaluate
criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

test_loss = total_loss / total
test_accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
