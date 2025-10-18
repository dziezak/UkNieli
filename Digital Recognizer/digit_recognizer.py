from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F

# jeśli model jest zapisany jako state_dict:
# model = SimpleNN()
# model.load_state_dict(torch.load('model_mnist.pth', map_location='cpu'))
# model.eval()

# transformacja zgodna z treningiem
preprocess = T.Compose([
    T.Resize((28,28), interpolation=Image.BILINEAR),
    T.ToTensor(),                      # daje tensor o zakresie [0,1]
    T.Normalize((0.5,), (0.5,))        # --> zakres ~[-1,1]
])

def load_and_prepare_image(path):
    img = Image.open(path).convert('L')  # konwertuj do grayscale
    # opcjonalnie: przytnij białe marginesy i dopasuj kontrast (proste heurystyki)
    # img = ImageOps.autocontrast(img)

    # Decide if we need to invert: MNIST has white digit on black background (digit bright)
    arr = np.array(img).astype(np.float32) / 255.0
    # jeśli średnia jasność obrazu jest wysoka => prawdopodobnie tło białe i kreska czarna -> inwersja
    if arr.mean() > 0.5:
        img = ImageOps.invert(img)

    tensor = preprocess(img)            # shape: [1, 28, 28]
    tensor = tensor.unsqueeze(0)        # shape: [1, 1, 28, 28] (batch dim)
    return tensor

def classify_image(path, model, device='cpu', topk=3):
    model.to(device)
    model.eval()
    x = load_and_prepare_image(path).to(device)
    with torch.no_grad():
        out = model(x)                  # logity
        probs = F.softmax(out, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_idxs  = top_idxs.cpu().numpy()[0]
    return list(zip(top_idxs.tolist(), top_probs.tolist()))

# użycie:
# results = classify_image('my_drawing.png', model)
# print("Top predictions (digit, probability):", results)
