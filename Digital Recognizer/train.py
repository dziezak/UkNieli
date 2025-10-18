# train.py
import torch
from model import SimpleNN
from data import get_mnist_loaders

def train_model(epochs=5, lr=0.01):
    train_loader, test_loader = get_mnist_loaders()
    model = SimpleNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # zapisanie wytrenowanego modelu
    torch.save(model.state_dict(), 'mnist_model.pth')
    return model
