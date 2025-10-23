# Install Folium once if you don’t have it
# !pip install folium geopandas

import geopandas as gpd
import folium

# Load shapefile
gdf = gpd.read_file("Dataset/경계/경계/(안전지도)소방서관할구역(GI_FIRE_A)/GI_FIRE_A_20220113.shp")

# Convert to WGS84 (latitude/longitude)
gdf = gdf.to_crs(epsg=4326)

# Find map center
center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]

# Create a Folium map
m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')

# Add shapefile layer to map
folium.GeoJson(
    gdf,
    name="Fire Station Zones",
    tooltip=folium.GeoJsonTooltip(fields=["WARD_NM"], aliases=["Fire Station:"]),
    style_function=lambda x: {"fillColor": "#%06x" % (hash(x['properties']['WARD_NM']) & 0xFFFFFF),
                              "color": "black", "weight": 1, "fillOpacity": 0.6}
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
m.save("fire_station_zones.html")
print("✅ Map saved as 'fire_station_zones.html' — open it in your browser!")