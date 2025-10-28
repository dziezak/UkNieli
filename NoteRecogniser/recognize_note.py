# recognize_note.py
import torch
from PIL import Image
from torchvision import transforms
from model import NoteCNN  # import modelu z model.py

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

# --- Funkcja rozpoznawania nuty ---
def recognize_note(img_path="my_note.png"):
    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0)  # dodajemy wymiar batch
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    print(f"🎵 Model rozpoznał nutę: {classes[pred.item()]}")

# --- Główna logika ---
if __name__ == "__main__":
    recognize_note("my_note.png")
    input("Naciśnij Enter, aby zamknąć aplikację...")
