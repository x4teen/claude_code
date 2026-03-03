"""
Original MNIST CNN — preserved for comparison.
Run directly or via compare_mnist_models.py.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

# --- Data ---
def get_data_loaders(transform, batch_size=64):
    train_data = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_data  = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size)
    return train_loader, test_loader

# --- Model ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def run(device=None, seed=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "generator") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])
    train_loader, test_loader = get_data_loaders(transform, BATCH_SIZE)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        n = len(train_loader.dataset)
        return running_loss / n, correct / n

    def evaluate():
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
        return correct / len(test_loader.dataset)

    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_epoch(epoch)
    train_time = time.perf_counter() - t0
    test_acc = evaluate()

    return {"test_acc": test_acc, "train_time_sec": train_time, "epochs": EPOCHS}


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training original model on {DEVICE}\n")
    result = run(device=DEVICE, seed=42)
    print(f"\nTest accuracy: {result['test_acc']:.2%}")
    print(f"Train time: {result['train_time_sec']:.1f}s")
    # Optionally save (uncomment if desired)
    # model = CNN().to(DEVICE)
    # torch.save(model.state_dict(), "mnist_cnn_original.pth")
