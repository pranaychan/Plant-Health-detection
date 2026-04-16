import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter
import numpy as np
import cv2
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder("PlantVillage")

    indices = torch.randperm(len(full_dataset))
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = datasets.ImageFolder("PlantVillage", transform=train_transform)
    val_dataset = datasets.ImageFolder("PlantVillage", transform=val_transform)

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    labels = [full_dataset.targets[i] for i in train_indices]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[i] for i in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": full_dataset.classes
            }, "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            break

        print(f"Stage1 Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    counter = 0

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": full_dataset.classes
            }, "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            break

        print(f"Stage2 Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    def gradcam(model, image_path, output_path):
        model.eval()

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(img_rgb).unsqueeze(0).to(device)

        gradients = []
        activations = []

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        def forward_hook(module, input, output):
            activations.append(output)

        target_layer = model.features[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1)

        model.zero_grad()
        output[0, pred_class].backward()

        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        cv2.imwrite(output_path, overlay)

    os.makedirs("gradcam_outputs", exist_ok=True)
    sample_path = full_dataset.samples[0][0]
    gradcam(model, sample_path, "gradcam_outputs/sample.jpg")

if __name__ == "__main__":
    main()