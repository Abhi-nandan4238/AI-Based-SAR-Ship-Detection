from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

image_path = next(RAW_DIR.glob("*.jpg"))

img = Image.open(image_path)

print("Image:", image_path.name)
print("Size:", img.size)
print("Mode:", img.mode)

plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title("Sentinel-1 SAR Visualization")
plt.axis("off")
plt.show()