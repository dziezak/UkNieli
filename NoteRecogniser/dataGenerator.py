from PIL import Image, ImageDraw
import random, os

def draw_note(position, save_path):
    img = Image.new('L', (100, 60), color=255)
    draw = ImageDraw.Draw(img)
    # pięciolinia
    for i in range(5):
        y = 20 + i*5
        draw.line((10, y, 90, y), fill=0, width=1)
    # pozycje nut (od dołu)
    positions = {
        "E4": 20+4*2.5,
        "F4": 20+3.5*2.5,
        "G4": 20+3*2.5,
        "A4": 20+2.5*2.5,
        "H4": 20+2*2.5,
        "C5": 20+1.5*2.5,
        "D5": 20+1*2.5,
    }
    y_note = positions[position]
    draw.ellipse((45, y_note-3, 55, y_note+3), fill=0)
    img.save(save_path)

os.makedirs("data/C5", exist_ok=True)
for i in range(100):
    draw_note("C5", f"data/C5/{i}.png")
