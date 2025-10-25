# draw_note.py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
from model import SimpleNoteNN
from torchvision import transforms
from PIL import Image

# --- Transformacja jak przy treningu ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((70, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Wczytanie modelu ---
classes = ["C4", "D4", "E4", "F4", "G4", "A4", "H4", "C5", "D5", "E5", "F5", "G5", "A5"]
model = SimpleNoteNN(num_classes=len(classes))
model.load_state_dict(torch.load("note_model.pth"))
model.eval()

# --- Funkcja rysowania nuty ---
def draw_note(save_path="my_note.png"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 70)

    # pięciolinia
    line_y = [20, 25, 30, 35, 40]
    for y in line_y:
        ax.plot([10, 110], [y, y], color="black", linewidth=1)

    ax.set_title("Kliknij w miejsce, gdzie chcesz postawić nutę")
    plt.ion()
    plt.show(block=False)
    coords = plt.ginput(1, timeout=-1)
    plt.close(fig)

    if len(coords) == 0:
        print("Nie narysowano żadnej nuty.")
        return None

    x, y = coords[0]

    # Tworzymy obraz dokładnie taki jak w dataset
    img = Image.new("L", (120, 70), 255)
    draw = ImageDraw.Draw(img)

    # linie pięciolinii
    for ly in line_y:
        draw.line((10, ly, 110, ly), fill=0, width=1)

    # elipsa nuty
    draw.ellipse((x-6, y-3, x+6, y+3), fill=0)

    # linie pomocnicze
    if y < 20:  # nuty powyżej 5 linii
        draw.line((x-10, 20, x+10, 20), fill=0, width=1)
    if y > 40:  # nuty poniżej 1 linii
        draw.line((x-10, 40, x+10, 40), fill=0, width=1)

    img.save(save_path)
    print(f"Nuta zapisana jako {save_path}")
    return save_path
