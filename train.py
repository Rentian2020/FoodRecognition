
import os
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Configuration
seed = 42
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 50
learning_rate = 0.001
weight_decay = 1e-4
patience = 7

selected_classes = [
    "apple_pie",
    "club_sandwich",
    "french_fries",
    "chicken_wings",
    "grilled_cheese_sandwich",
    "hamburger",
    "hot_dog",
    "pancakes",
    "pizza",
    "steak"
]

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Using device:", device)

# Image preprocessing
weights = MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = weights.transforms()

# Load Food-101
train_data = Food101(
    root="./data",
    split="train",
    transform=train_transform,
    download=False
)

val_data = Food101(
    root="./data",
    split="train",
    transform=eval_transform,
    download=False
)

test_data = Food101(
    root="./data",
    split="test",
    transform=eval_transform,
    download=False
)

# Keep only selected classes
class_to_new_idx = {
    train_data.class_to_idx[name]: idx
    for idx, name in enumerate(selected_classes)
}


class FoodSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, old_label = self.dataset[self.indices[index]]
        return image, class_to_new_idx[old_label]


# Stratified train/validation split
train_indices = []
val_indices = []

rng = random.Random(seed)

for class_name in selected_classes:
    original_label = train_data.class_to_idx[class_name]

    indices = [
        i for i, label in enumerate(train_data._labels)
        if label == original_label
    ]

    rng.shuffle(indices)

    split = int(len(indices) * 0.8)

    train_indices.extend(indices[:split])
    val_indices.extend(indices[split:])

test_indices = [
    i for i, label in enumerate(test_data._labels)
    if label in class_to_new_idx
]

train_subset = FoodSubset(train_data, train_indices)
val_subset = FoodSubset(val_data, val_indices)
test_subset = FoodSubset(test_data, test_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_subset))
print("Validation images:", len(val_subset))
print("Test images:", len(test_subset))

# Load pretrained MobileNetV3-Small
model = mobilenet_v3_small(weights=weights)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace final classification layer
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(selected_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay
)


# Evaluation
def evaluate(loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


# Training
history = []
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

model_path = "models/mobilenetv3_food_v2.pth"

for epoch in range(epochs):
    model.train()

    # Keep frozen feature extractor in eval mode
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100 * correct / total

    val_loss, val_acc = evaluate(val_loader)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": selected_classes,
            "epoch": best_epoch,
            "val_loss": best_val_loss
        }, model_path)

        print("  Best model saved.")

    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}.")
        break


# Load best model
checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Final test evaluation
test_loss, test_acc = evaluate(test_loader)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Model saved to {model_path}")

# Save training history
history_path = "results/training_history_v2.csv"

with open(history_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(history)

print(f"Training history saved to {history_path}")
