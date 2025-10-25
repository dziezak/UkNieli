# model.py
import torch
from torch import nn
import torch.nn.functional as F

class NoteCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Poprawny wymiar po pooling
        self.fc1 = nn.Linear(32 * 35 * 60, 128)  # 32 kanały, 35x60 po poolingu
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # spłaszczamy
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
