import os
import geopandas as gpd
import matplotlib.pyplot as plt

try:
	import pandas as pd
except Exception:
	raise SystemExit("pandas is required. Please install dependencies in your environment.")


# Edit this path to the shapefile you want to read
SHP_PATH = r"Dataset/경계/경계/통계청_SGIS 행정구역 통계 및 경계_20240630/2. 경계/1. 2024년 2분기 기준 시도 경계/bnd_sido_00_2024_2Q.shp"

# Optional output
CSV_OUT = False

# Plot after loading
PLOT = False

def read_shapefile(shp_path: str) -> gpd.GeoDataFrame:
	if not os.path.exists(shp_path):
		raise FileNotFoundError(f"Shapefile not found: {shp_path}")
	gdf = gpd.read_file(shp_path)
	return gdf


def export_attributes(gdf: gpd.GeoDataFrame, csv_path: str | None) -> None:
	# Convert geometry to WKT so it can be saved in tabular formats
	df = gdf.copy()
	df["geometry_wkt"] = df.geometry.to_wkt()
	df_no_geom = df.drop(columns=["geometry"]) if "geometry" in df.columns else df

	if csv_path:
		os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
		df_no_geom.to_csv(csv_path, index=False)
		print(f"✓ Saved CSV:   {csv_path}")


def plot_gdf(gdf: gpd.GeoDataFrame, title: str) -> None:
	ax = gdf.plot(edgecolor="black", facecolor="lightcoral", figsize=(10, 8))
	ax.set_title(title, fontsize=14)
	ax.set_xlabel("Longitude / X")
	ax.set_ylabel("Latitude / Y")
	plt.tight_layout()
	plt.show()


def main():
	shp_path = SHP_PATH

	# Read
	gdf = read_shapefile(shp_path)

	# Summary
	print("\n=== Layer summary ===")
	print(f"Path: {shp_path}")
	print(f"Rows: {len(gdf):,}")
	print(f"Columns ({len(gdf.columns)}): {list(gdf.columns)}")
	print(f"CRS: {gdf.crs}")

	# Show first rows just for quick look; the full dataset will be exported
	print("\n=== Head (first 5 rows) ===")
	print(gdf.head())

	# Export
	# Save to project root folder
	base_name = os.path.splitext(os.path.basename(shp_path))[0]
	csv_out = CSV_OUT or f"{base_name}.csv"

	export_attributes(gdf, csv_out)

	# Plot
	if PLOT:
		title = f"{base_name} (CRS: {gdf.crs.to_string() if gdf.crs else 'None'})"
		plot_gdf(gdf, title)


if __name__ == "__main__":
	main()