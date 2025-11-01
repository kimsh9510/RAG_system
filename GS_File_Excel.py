# 1️⃣ Install required libraries (only once)
# !pip install geopandas matplotlib

# 2️⃣ Import libraries
import geopandas as gpd
import matplotlib.pyplot as plt

# 3️⃣ Read your shapefile
# Replace the path below with the actual folder path where your files are saved
gdf = gpd.read_file(r"Dataset/경계/경계/(안전지도)소방서관할구역(GI_FIRE_A)/GI_FIRE_A_20220113.shp")

# 4️⃣ Display the first few rows of attribute data (from .dbf)
print(gdf.head())

# 5️⃣ Check Coordinate Reference System (from .prj)
print("\nCoordinate Reference System (CRS):", gdf.crs)

# 6️⃣ Plot the map visually
gdf.plot(edgecolor='black', facecolor='lightcoral', figsize=(10,8))
plt.title("Fire Station Jurisdiction Areas", fontsize=14)
plt.xlabel("Longitude / X")
plt.ylabel("Latitude / Y")
plt.show()