# import statements
import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
from flask import Flask, render_template, request, jsonify, send_file
import geopandas as gpd
from shapely.geometry import Point, box
import json
import requests
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)


def geocode(address):
    # Nominatim is OpenStreetMap's free geocoding API
    url = "https://nominatim.openstreetmap.org/search"

    # try/except catches network errors or bad responses so the app does not crash
    try:
        # send the address as a query, ask for JSON back
        r = requests.get(url, params={"q": address, "format": "json"}, headers={"User-Agent": "salmonshield"})
        results = r.json()

        # if the API returns an empty list, the address was not found
        if not results:
            return None

        # grab the first result (most relevant match)
        result = results[0]

        # return as lat, lon floats
        return float(result["lat"]), float(result["lon"])
    except Exception:
        # returns None on any failure (network error, bad JSON, etc.)
        return None



# load geoJSON datasets
def load_geo(path, crs="EPSG:4326"):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    try:
        return gpd.read_file(path)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

streams = load_geo("data/salmon_streams.geojson")
stormwater = load_geo("data/storm_discharge.geojson")
watersheds = load_geo("data/watershed_boundaries.geojson")
water_quality = load_geo("data/water_quality_pollutants.geojson")
heavy_traffic = load_geo("data/proximity_heavy_traffic_roadways.geojson")

@app.route("/")
def home():
    return render_template("index.html")

# route that takes an address string and returns lat/lon coordinates as JSON
@app.route("/geocode", methods=["POST"])
def geocode_address():
    # get address from the form data
    address = request.form.get("address")
    # return 400 error if no address was sent
    if not address:
        return jsonify({"error": "No address provided"}), 400

    # convert the address to coordinates using the geocode function
    coords = geocode(address)
    # return 404 if the address could not be found
    if not coords:
        return jsonify({"error": "Could not find that address"}), 404

    return jsonify({"lat": coords[0], "lon": coords[1]})

# risk color based on score
def score_to_color(score):
    if score >= 60:
        return "red"
    elif score >= 30:
        return "yellow"
    else:
        return "green"


# per-stream risk color based on its distance and nearby drain count
def get_risk_color(distance_m, drain_count):
    proximity = 70 * max(0, 1 - distance_m / 1000)
    drain_boost = min(30, drain_count * 6)
    score = max(0, min(100, int(proximity + drain_boost)))
    return score_to_color(score)

def _geodf_to_features(gdf):
    """Convert a GeoDataFrame to GeoJSON features, handling Timestamp columns."""
    for col in gdf.select_dtypes(include=["datetime", "datetimetz"]).columns:
        gdf[col] = gdf[col].astype(str)
    return json.loads(gdf.to_json())["features"]

# runs the analysis for a given lat/lon, returns a dictionary of results
def run_analysis(lat, lon):
    # make a point from the coordinates
    user_point = Point(lon, lat)

    # convert to meters so we can measure distances accurately
    user_point_m = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    # if no data loaded, return empty defaults
    if streams.empty and stormwater.empty:
        return {"nearby_streams": [], "nearby_stormwater": [], "impact_score": 0, "risk_color": "green"}

    # convert datasets to meters (only if they have data)
    streams_m = streams.to_crs(epsg=3857) if not streams.empty else streams
    stormwater_m = stormwater.to_crs(epsg=3857) if not stormwater.empty else stormwater

    # find streams and stormwater within 1 km
    nearby_streams = streams_m[streams_m.geometry.distance(user_point_m) < 1000] if not streams_m.empty else streams_m
    nearby_stormwater = stormwater_m[stormwater_m.geometry.distance(user_point_m) < 1000] if not stormwater_m.empty else stormwater_m

    # distance to the closest stream (default 1km if none found)
    nearest_distance = nearby_streams.geometry.distance(user_point_m).min() if not nearby_streams.empty else 1000

    # --- impact score: 3 balanced components (0-100) ---
    # proximity: closer to a stream = higher impact (max 40 pts)
    proximity_score = 40 * max(0, 1 - nearest_distance / 1000)
    # stream density: more streams nearby = more habitat at risk (max 30 pts)
    density_score = min(30, len(nearby_streams) * 5)
    # stormwater: more drains = more pollution sources (max 30 pts)
    stormwater_score = min(30, len(nearby_stormwater) * 5)
    impact_score = max(0, min(100, int(proximity_score + density_score + stormwater_score)))

    # color each stream by its own distance-based risk
    nearby_streams = nearby_streams.copy()
    drain_count = len(nearby_stormwater)
    nearby_streams['riskColor'] = [
        get_risk_color(geom.distance(user_point_m), drain_count)
        for geom in nearby_streams.geometry
    ]

    # if no streams within 1 km, find the closest one anywhere
    nearest_stream_point = None
    nearest_stream_dist_km = None
    nearest_stream_feature = None
    if nearby_streams.empty and not streams_m.empty:
        all_distances = streams_m.geometry.distance(user_point_m)
        closest_idx = all_distances.idxmin()
        nearest_stream_dist_km = round(all_distances[closest_idx] / 1000, 1)
        # nearest point on that stream
        closest_geom = streams_m.geometry[closest_idx]
        nearest_pt_m = closest_geom.interpolate(closest_geom.project(user_point_m))
        nearest_pt_4326 = gpd.GeoSeries([nearest_pt_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        nearest_stream_point = [nearest_pt_4326.y, nearest_pt_4326.x]
        # also grab the full stream geometry so it can be drawn
        row = streams.loc[closest_idx]
        # find which watershed this stream falls in
        ws_name = None
        if not watersheds.empty:
            rep_pt = row.geometry.representative_point()
            containing = watersheds[watersheds.contains(rep_pt)]
            if not containing.empty:
                ws_name = containing.iloc[0].get("Name", None)
        nearest_stream_feature = {
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "LLID_STRM_NAME": row.get("LLID_STRM_NAME", "Nearest stream"),
                "riskColor": get_risk_color(all_distances[closest_idx], drain_count),
                "watershed_name": ws_name,
            }
        }

    # convert streams back to lat/lon for Leaflet (Leaflet uses EPSG:4326)
    nearby_streams_4326 = nearby_streams.to_crs(epsg=4326) if not nearby_streams.empty else nearby_streams
    raw_features = nearby_streams_4326.__geo_interface__['features'] if not nearby_streams_4326.empty else []

    # spatially join nearby streams to watersheds for watershed_name
    if not nearby_streams_4326.empty and not watersheds.empty:
        pts = nearby_streams_4326.copy()
        pts["_rep"] = pts.geometry.representative_point()
        pts = pts.set_geometry("_rep")
        joined = gpd.sjoin(pts, watersheds[["Name", "geometry"]], how="left", predicate="within")
        # map LLID -> watershed name (first match)
        ws_lookup = dict(zip(joined["LLID"], joined["Name"]))
    else:
        ws_lookup = {}

    # group features by stream ID (LLID) so each stream is drawn once
    seen = {}
    merged_features = []
    for f in raw_features:
        llid = f["properties"].get("LLID", "")
        sp = f["properties"].get("SPECIES", "")
        coord_count = len(f["geometry"]["coordinates"])

        if llid in seen:
            if sp and sp not in seen[llid]["properties"]["allSpecies"]:
                seen[llid]["properties"]["allSpecies"].append(sp)
            if coord_count > len(seen[llid]["geometry"]["coordinates"]):
                seen[llid]["geometry"] = f["geometry"]
        else:
            props = dict(f["properties"])
            props["allSpecies"] = [sp] if sp else []
            # attach watershed name from spatial join
            props["watershed_name"] = ws_lookup.get(llid, None)
            f["properties"] = props
            seen[llid] = f
            merged_features.append(f)

    # find watersheds intersecting a ~3km bbox around the user
    nearby_watersheds = []
    if not watersheds.empty:
        buf = 0.03  # ~3km in degrees
        bbox = box(lon - buf, lat - buf, lon + buf, lat + buf)
        hits = watersheds[watersheds.intersects(bbox)]
        if not hits.empty:
            # simplify geometry to reduce payload size
            hits_simple = hits.copy()
            hits_simple["geometry"] = hits_simple.geometry.simplify(0.0005)
            cols = [c for c in ["Name", "HUC12", "AreaSqKm", "geometry"] if c in hits_simple.columns]
            nearby_watersheds = json.loads(hits_simple[cols].to_json(default=str))["features"]

    # --- water quality: find the census tract(s) intersecting a ~500m buffer ---
    wq_data = None
    if not water_quality.empty:
        buf_deg = 0.0045  # ~500 m in degrees latitude
        bbox = box(lon - buf_deg, lat - buf_deg, lon + buf_deg, lat + buf_deg)
        hits = water_quality[water_quality.intersects(bbox)]
        if not hits.empty:
            # prefer the tract that actually contains the point; fall back to bbox hits
            containing = hits[hits.contains(user_point)]
            tracts = containing if not containing.empty else hits
            total_impairments = int(tracts["TotalUniqueImpairments"].sum())
            max_rank = int(tracts["Water_Quality_Rank"].max())
            wq_data = {
                "total_impairments": total_impairments,
                "max_rank": max_rank,
                "tract_count": len(tracts)
            }

    # --- heavy traffic: find the census tract(s) intersecting a ~1km buffer ---
    traffic_data = None
    if not heavy_traffic.empty:
        buf_deg = 0.009  # ~1 km in degrees latitude
        bbox = box(lon - buf_deg, lat - buf_deg, lon + buf_deg, lat + buf_deg)
        hits = heavy_traffic[heavy_traffic.intersects(bbox)]
        if not hits.empty:
            containing = hits[hits.contains(user_point)]
            tracts = containing if not containing.empty else hits
            traffic_data = {
                "ehd_rank": int(tracts["EHD_Rank"].max()),
                "env_exp_rank": int(tracts["Env_Exp_Rank"].max()),
            }

    return {
        "nearby_streams": merged_features,
        "nearby_stormwater": _geodf_to_features(nearby_stormwater.to_crs(epsg=4326)) if not nearby_stormwater.empty else [],
        "nearby_watersheds": nearby_watersheds,
        "impact_score": impact_score,
        "risk_color": score_to_color(impact_score),
        # nearest stream info (only set when no streams are within 1 km)
        "nearest_stream_point": nearest_stream_point,
        "nearest_stream_dist_km": nearest_stream_dist_km,
        "nearest_stream_feature": nearest_stream_feature,
        "water_quality": wq_data,
        "traffic": traffic_data
    }


# API endpoint — returns JSON (kept for any future use)
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        lat = float(request.form["latitude"])
        lon = float(request.form["longitude"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid or missing coordinates"}), 400

    return jsonify(run_analysis(lat, lon))


# results page — runs analysis and renders the HTML template
@app.route("/results")
def results():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return "Missing or invalid lat/lon", 400

    # run the analysis
    data = run_analysis(lat, lon)

    # pass everything to the template
    return render_template("results.html",
        lat=lat,
        lon=lon,
        streams_json=json.dumps(data["nearby_streams"]),
        stormwater_json=json.dumps(data["nearby_stormwater"]),
        watersheds_json=json.dumps(data["nearby_watersheds"]),
        impact_score=data["impact_score"],
        risk_color=data["risk_color"],
        nearest_stream_point=json.dumps(data["nearest_stream_point"]),
        nearest_stream_feature_json=json.dumps(data["nearest_stream_feature"]),
        nearest_stream_dist_km=data["nearest_stream_dist_km"],
        water_quality_json=json.dumps(data["water_quality"]),
        traffic_json=json.dumps(data["traffic"])
    )

# --- NLCD impervious surface tile endpoint ---
# path to the 2024 fractional impervious surface GeoTIFF
NLCD_TIFF = "data/NLCD_b812e669-a6c9-44dd-a6b7-127872e98e5c/Annual_NLCD_FctImp_2024_CU_C1V1_b812e669-a6c9-44dd-a6b7-127872e98e5c.tiff"

# color ramp: 0% impervious = transparent, 100% = dark red
# matches standard NLCD impervious palette
def impervious_to_rgba(band):
    """Convert a uint8 impervious % array to an RGBA image array."""
    h, w = band.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = band > 0  # only color pixels with some impervious cover
    pct = band[mask].astype(float) / 100.0
    # gradient: green(low) -> yellow(mid) -> red(high)
    rgba[mask, 0] = (pct * 255).astype(np.uint8)           # R increases
    rgba[mask, 1] = ((1 - pct) * 200).astype(np.uint8)     # G decreases
    rgba[mask, 2] = 30                                       # B constant
    rgba[mask, 3] = (60 + pct * 140).astype(np.uint8)       # alpha: 60-200
    return rgba

@app.route("/nlcd_tile")
def nlcd_tile():
    """Serve a reprojected PNG overlay of NLCD impervious surface.
    Query params: west, south, east, north (EPSG:4326), width, height in px.
    Reprojects from the source Albers CRS to EPSG:4326 so it aligns with Leaflet.
    """
    try:
        west = float(request.args["west"])
        south = float(request.args["south"])
        east = float(request.args["east"])
        north = float(request.args["north"])
        width = int(request.args.get("width", 512))
        height = int(request.args.get("height", 512))
    except (KeyError, ValueError):
        return "Missing bbox params", 400

    # convert 4326 bbox to 3857 (Web Mercator) — matches Leaflet's internal projection
    merc_bounds = transform_bounds("EPSG:4326", "EPSG:3857", west, south, east, north)
    dst_transform = from_bounds(*merc_bounds, width, height)
    dst_array = np.zeros((height, width), dtype=np.uint8)

    with rasterio.open(NLCD_TIFF) as src:
        # reproject source raster into 3857 grid so pixels align with Leaflet
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.bilinear,
        )

    # convert to colored RGBA PNG
    rgba = impervious_to_rgba(dst_array)
    img = Image.fromarray(rgba, "RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# allows running the app directly with "python app.py" instead of "flask run"
if __name__ == "__main__":
    app.run(debug=True, port=5000)
