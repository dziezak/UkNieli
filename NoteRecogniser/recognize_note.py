# recognize_note.py
import matplotlib
matplotlib.use("TkAgg")  # interaktywny backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import torch
from torchvision import transforms
from model import NoteCNN

# --- Klasy nut (w tej samej kolejności co przy treningu) ---
classes = ["C4", "D4", "E4", "F4", "G4", "A4", "H4", "C5", "D5", "E5", "F5", "G5", "A5"]

# --- Wczytanie modelu ---
model = NoteCNN(num_classes=len(classes))
model.load_state_dict(torch.load("note_model.pth"))
model.eval()

# --- Transformacje identyczne jak przy treningu ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((70, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Funkcja rysowania nuty ---
def draw_note(save_path="my_note.png"):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # rysujemy pięciolinię
    for i in range(5):
        ax.plot([0, 10], [3 + i, 3 + i], color="black", linewidth=1)

    ax.set_title("Kliknij w miejsce, gdzie chcesz postawić nutę")
    plt.ion()
    plt.show(block=False)
    coords = plt.ginput(1, timeout=-1)  # czekamy na kliknięcie
    plt.close(fig)

    if len(coords) == 0:
        print("Nie narysowano żadnej nuty.")
        return None

    x, y = coords[0]

    # Tworzymy obraz z poziomą elipsą nuty
    fig, ax = plt.subplots(figsize=(5, 3))
    for i in range(5):
        ax.plot([0, 10], [3 + i, 3 + i], color="black", linewidth=1)
    ellipse = Ellipse((x, y), width=2.5, height=1.5, color="black")  # bardziej pozioma
    ax.add_patch(ellipse)
    ax.axis("off")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show(block=True)

    print(f"Nuta zapisana jako {save_path}")
    return save_path

# --- Funkcja rozpoznawania nuty ---
def recognize_note(img_path):
    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    print(f"🎵 Model rozpoznał nutę: {classes[pred.item()]}")

# --- Główna logika ---
if __name__ == "__main__":
    path = draw_note()
    if path:
        recognize_note(path)
        input("Naciśnij Enter, aby zamknąć aplikację...")
