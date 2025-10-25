# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import NoteCNN

# --- Przygotowanie danych ---
data_dir = "./NoteRecogniser/data_notes_aligned"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((70, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = dataset.classes
print("Klasy:", classes)

# --- Model ---
model = NoteCNN(num_classes=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Trening ---
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoka {epoch+1}/{epochs}, Strata: {total_loss/len(train_loader):.4f}")

# --- Testowanie ---
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Dokładność testu: {100 * correct / total:.2f}%")

# --- Zapis wytrenowanego modelu ---
torch.save(model.state_dict(), "note_model.pth")
print("✅ Model zapisany jako note_model.pth")
