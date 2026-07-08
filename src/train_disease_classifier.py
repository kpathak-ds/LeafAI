# src/train_disease_classifier.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Setup device (MPS for mac, CUDA for nvidia, CPU fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Load Dataset ──────────────────────────────────────────
class CropDiseaseDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data_from_folders():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    olid_dir = os.path.join(base_dir, "olid")

    if not os.path.exists(olid_dir):
        raise FileNotFoundError(f"olid folder not found at: {olid_dir}")

    images = []
    labels = []
    
    # 5-class mapping:
    # 0: Snake Gourd Healthy
    # 1: Snake Gourd Nitrogen Deficiency
    # 2: Tomato Healthy
    # 3: Tomato Nitrogen Deficiency
    # 4: Tomato Potassium Deficiency
    class_mapping = {
        "snake_gourd__healthy": 0,
        "snake_gourd__N": 1,
        "tomato__healthy": 2,
        "tomato__N": 3,
        "tomato__K": 4
    }

    for folder_name in os.listdir(olid_dir):
        folder_path = os.path.join(olid_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = class_mapping.get(folder_name)
        if label is None:
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def train_model():
    X, y = load_data_from_folders()
    print(f"Loaded {len(X)} images. Classes distribution:")
    for label_id in range(5):
        count = np.sum(y == label_id)
        print(f"  Class {label_id}: {count} images")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CropDiseaseDataset(X_train, y_train, train_transform)
    val_dataset = CropDiseaseDataset(X_val, y_val, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ── Load EfficientNet-B0 ──────────────────────────────────
    print("Loading pre-trained EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Replace classifier head to output 5 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print("Training disease and nutrient deficiency classifier...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc*100:.2f}% - Val Acc: {val_acc*100:.2f}%")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "disease_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved successfully at: {save_path}")

if __name__ == "__main__":
    train_model()
