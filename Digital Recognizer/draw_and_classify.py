# draw_and_classify.py
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from model import SimpleNN

# transformacja zgodna z MNIST
preprocess = T.Compose([
    T.Resize((28,28), interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def load_and_prepare_image(path):
    img = Image.open(path).convert('L')  # grayscale

    # inwersja jeśli tło jasne
    arr = np.array(img).astype(np.float32)/255.0
    if arr.mean() > 0.5:
        img = ImageOps.invert(img)

    tensor = preprocess(img)         # [1,28,28]
    tensor = tensor.unsqueeze(0)     # [1,1,28,28] batch
    return tensor

def classify_image(path, model, device='cpu', topk=3):
    model.to(device)
    model.eval()
    x = load_and_prepare_image(path).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_idxs  = top_idxs.cpu().numpy()[0]
    return list(zip(top_idxs.tolist(), top_probs.tolist()))

def run_draw_app(model, save_path='draw.png'):
    width, height = 280, 280
    root = tk.Tk()
    root.title("Draw a digit and press Classify")

    canvas = tk.Canvas(root, width=width, height=height, bg='white')
    canvas.pack()

    image1 = Image.new("L", (width, height), 'white')
    draw = ImageDraw.Draw(image1)

    last_x, last_y = None, None
    brush_size = 18

    def xy(event):
        nonlocal last_x, last_y
        x, y = event.x, event.y
        if last_x is not None and last_y is not None:
            canvas.create_line(last_x, last_y, x, y, width=brush_size, fill='black', capstyle=tk.ROUND, smooth=True)
            draw.line([last_x, last_y, x, y], fill='black', width=brush_size)
        last_x, last_y = x, y

    def release(event):
        nonlocal last_x, last_y
        last_x, last_y = None, None

    def clear():
        canvas.delete("all")
        draw.rectangle([0,0,width,height], fill='white')

    def classify_and_show():
        image1.save(save_path)
        results = classify_image(save_path, model)
        print("Top predictions:", results)
        label_var.set(f"Top: {results[0][0]} ({results[0][1]*100:.1f}%)")

    canvas.bind("<B1-Motion>", xy)
    canvas.bind("<ButtonRelease-1>", release)

    frame = tk.Frame(root)
    frame.pack()
    btn_classify = tk.Button(frame, text="Classify", command=classify_and_show)
    btn_classify.pack(side=tk.LEFT)
    btn_clear = tk.Button(frame, text="Clear", command=clear)
    btn_clear.pack(side=tk.LEFT)

    label_var = tk.StringVar()
    label = tk.Label(root, textvariable=label_var)
    label.pack()

    root.mainloop()
