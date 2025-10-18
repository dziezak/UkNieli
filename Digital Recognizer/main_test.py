# main_test.py
from model import SimpleNN
from test_and_visualize import test_model, visualize_predictions
import torch

model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

test_model(model)
visualize_predictions(model)
