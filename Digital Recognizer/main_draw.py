# main_draw.py
from model import SimpleNN
from draw_and_classify import run_draw_app
import torch

# wczytujemy wytrenowany model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# uruchamiamy rysownik
run_draw_app(model)
