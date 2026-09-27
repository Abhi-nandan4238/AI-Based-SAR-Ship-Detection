import rasterio
from rasterio.env import Env

print("Rasterio Version:", rasterio.__version__)

with Env() as env:
    drivers = env.drivers()

    print("Total Drivers:", len(drivers))
    print("GTiff:", "GTiff" in drivers)
    print("JP2OpenJPEG:", "JP2OpenJPEG" in drivers)