import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transformacje: konwertujemy obraz na tensor i normalizujemy
transform = transforms.Compose([
    transforms.ToTensor(),             # zmienia obraz 28x28 w tensor
    transforms.Normalize((0.5,), (0.5,))  # normalizacja na zakres [-1, 1]
])

# Wczytanie zbioru treningowego i testowego
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Tworzymy DataLoader – pozwala iterować po batchach
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 3. Definicja modelu
    # podstawowa klasa dla PyTorch to nn.Module
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) # tworzymy sobie 128 neuronów z 28*28 klatek
        self.fc2 = nn.Linear(128, 10) # a później tworzymy z 128 neuroów 10 neuronów ( czyli 0-9)
    def forward(self, x):
        x = x.view(-1, 28*28) # zamieniamy na jeden wektor
        # każdy z pikleli jest ważony przez wagę i dodawany do biasu. Matematycznie: y = W*x + b
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # tutaj 128 -> 10 neuronow
        return x
# x (784 wejść) ---> fc1 ---> 128 neuronów ---> ReLU ---> 128 wartości po aktywacji


model = SimpleNN()

# 4. Funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss() # porównujemy faktyczne wyniki
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # aktualizacja wag modelu


#5 Trening modelu
epochs = 5  # liczba przejść przez cały zbiór treningowy

for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()       # zerujemy gradienty
        outputs = model(images)     # przewidywania modelu
        loss = criterion(outputs, labels)  # obliczamy błąd
        loss.backward()             # backpropagation
        optimizer.step()            # aktualizacja wag
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

#6 testowanie modelu
correct = 0
total = 0
with torch.no_grad():  # wyłączamy gradienty
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # wybieramy klasę z największym prawdopodobieństwem
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


#7 wyswietlanie przykladowych predykcji:
import matplotlib.pyplot as plt

images, labels = next(iter(test_loader))
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i].view(28,28), cmap='gray')
    plt.title(f"P: {preds[i].item()} / T: {labels[i].item()}")
    plt.axis('off')
plt.show()

digit_recognizer.py
