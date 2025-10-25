# test_random_notes.py
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# 🔹 Folder z danymi
base_dir = "./NoteRecogniser/data_notes_aligned"

# 🔹 Klasy nut
classes = ["C4", "D4", "E4", "F4", "G4", "A4", "H4", "C5", "D5", "E5", "F5", "G5", "A5"]

# 🔹 Ilość losowych przykładów do podglądu
samples_per_class = 2

# 🔹 Lista obrazów i etykiet
images = []
labels = []

for cls in classes:
    folder = os.path.join(base_dir, cls)
    if not os.path.exists(folder):
        print(f"⚠️ Folder nie istnieje: {folder}")
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    if len(files) == 0:
        print(f"⚠️ Brak plików w folderze: {folder}")
        continue
    selected_files = random.sample(files, min(samples_per_class, len(files)))
    for f in selected_files:
        images.append(Image.open(os.path.join(folder, f)).convert("L"))
        labels.append(cls)

# 🔹 Wyświetlanie obrazów
cols = 6
rows = (len(images) + cols - 1) // cols
plt.figure(figsize=(cols*1.6, rows*1.6))
for i, (im, lbl) in enumerate(zip(images, labels)):
    plt.subplot(rows, cols, i+1)
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.title(lbl)
    plt.axis("off")
plt.tight_layout()
plt.show()
