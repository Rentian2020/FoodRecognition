
import os
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# Configuration
seed = 42
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 30
learning_rate = 1e-5
weight_decay = 1e-4
patience = 7

source_model = "models/mobilenetv3_food_v2.pth"
output_model = "models/mobilenetv3_food_v3.pth"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Using device:", device)

# Load previous best checkpoint
checkpoint = torch.load(
    source_model,
    map_location="cpu",
    weights_only=True
)

selected_classes = checkpoint["classes"]

# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Same validation preprocessing as V2
from torchvision.models import MobileNet_V3_Small_Weights
eval_transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

# Load datasets
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

# Reproduce the same train/validation split as V2
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

train_subset = FoodSubset(train_data, train_indices)
val_subset = FoodSubset(val_data, val_indices)

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

print("Training images:", len(train_subset))
print("Validation images:", len(val_subset))

# Build model and restore V2 weights
model = mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(selected_classes)
)

model.load_state_dict(checkpoint["model_state_dict"])

# Freeze all feature blocks first
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last 3 feature blocks
for block in model.features[-3:]:
    for param in block.parameters():
        param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay
)


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


# Fine-tuning
history = []
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()

    # Keep frozen blocks and BatchNorm statistics fixed
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": selected_classes,
            "epoch": best_epoch,
            "val_loss": best_val_loss
        }, output_model)

        print("  Best fine-tuned model saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}.")
        break

# Save history
history_path = "results/finetuning_history_v3.csv"

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

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to {output_model}")
print(f"Training history saved to {history_path}")
