"""Create sample test images for inference benchmarking."""
from PIL import Image
import os

out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "images")
os.makedirs(out_dir, exist_ok=True)

sizes = [(224, 224), (640, 480), (1920, 1080)]
for idx, (w, h) in enumerate(sizes, 1):
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = (
                (x * 7 + y * 3) % 256,
                (x * 11 + y * 5) % 256,
                (x * 13 + y * 7) % 256,
            )
    path = os.path.join(out_dir, f"test_{idx}_{w}x{h}.jpg")
    img.save(path)
    print(f"Created {os.path.basename(path)} ({os.path.getsize(path)} bytes)")
