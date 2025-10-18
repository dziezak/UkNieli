# test_and_visualize.py
import torch
import matplotlib.pyplot as plt
from model import SimpleNN
from data import get_mnist_loaders

def test_model(model):
    _, test_loader = get_mnist_loaders()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def visualize_predictions(model, num_images=8):
    _, test_loader = get_mnist_loaders()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10,4))
    for i in range(num_images):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i].view(28,28), cmap='gray')
        plt.title(f"P: {preds[i].item()} / T: {labels[i].item()}")
        plt.axis('off')
    plt.show()
