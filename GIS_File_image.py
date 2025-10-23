import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Configure Korean font
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# Load your shapefile (already working in your setup)
gdf = gpd.read_file("Dataset/경계/경계/(안전지도)소방서관할구역(GI_FIRE_A)/GI_FIRE_A_20220113.shp", encoding="cp949")

# Convert CRS to WGS84 (EPSG:4326) so it matches world maps if needed
gdf = gdf.to_crs(epsg=4326)

# Plot each fire station area in a different color
fig, ax = plt.subplots(figsize=(10,8))
gdf.plot(column="WARD_NM", cmap="tab20", legend=True, ax=ax, edgecolor="black")

# Add title and labels
ax.set_title("Fire Station Jurisdiction Areas (소방서 관할구역)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Save the plot
plt.savefig("fire_stations_map.png", dpi=300, bbox_inches='tight')
print("Map saved as 'fire_stations_map.png'")

plt.show()