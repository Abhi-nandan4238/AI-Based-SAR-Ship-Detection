from pathlib import Path
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAR_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel1_grd"

files = list(SAR_DIR.rglob("*.tif")) + list(SAR_DIR.rglob("*.tiff"))

if not files:
    print("No GeoTIFF found.")
    print("SAR product may still be in SAFE/raw format.")
    raise SystemExit

sar_file = files[0]

print("SAR file:", sar_file.name)

with rasterio.open(sar_file) as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Data type:", src.dtypes[0])