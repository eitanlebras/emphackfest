"""Add watershed Name to each salmon stream via spatial join."""
import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
import geopandas as gpd

DATA = "/Users/vivaankansal/Documents/GitHub/emphackfest/data"

print("Loading watershed boundaries...")
ws = gpd.read_file(f"{DATA}/watershed_boundaries.geojson")[["Name", "geometry"]]
print(f"  {len(ws)} watersheds loaded")

print("Loading salmon streams...")
streams = gpd.read_file(f"{DATA}/salmon_streams.geojson")
print(f"  {len(streams)} streams loaded")

# Ensure matching CRS
if streams.crs != ws.crs:
    ws = ws.to_crs(streams.crs)

# Use representative point of each stream for the join
streams["_rep_point"] = streams.geometry.representative_point()
points = streams.set_geometry("_rep_point")

print("Spatial join...")
joined = gpd.sjoin(points, ws, how="left", predicate="within")

# Handle duplicates from overlapping polygons - keep first match
joined = joined[~joined.index.duplicated(keep="first")]

# Add watershed name back to streams
streams["watershed_name"] = joined["Name"]
streams.drop(columns=["_rep_point"], inplace=True, errors="ignore")

print("Writing output...")
streams.to_file(f"{DATA}/salmon_streams.geojson", driver="GeoJSON")

filled = streams["watershed_name"].notna().sum()
print(f"Done. {filled}/{len(streams)} streams matched to a watershed.")
