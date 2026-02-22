# --- setup & imports ---
import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # allow reading very large GeoJSON files
from dotenv import load_dotenv
load_dotenv()  # load API keys from .env file
from flask import Flask, render_template, request, jsonify, send_file
import geopandas as gpd        # geospatial dataframes (like pandas but for maps)
from shapely.geometry import Point, box  # geometric shapes for spatial queries
import json
import requests                # HTTP requests to external APIs
import openai                  # GPT-4o for generating suggestions
import rasterio                # reading raster/satellite imagery files (GeoTIFF)
from rasterio.warp import transform_bounds, reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import numpy as np             # number crunching for image arrays
from io import BytesIO         # in-memory file buffer for PNG images
from PIL import Image          # image creation/manipulation

app = Flask(__name__)  # create the Flask web app


# convert an address string to (lat, lon) using OpenStreetMap
def geocode(address):
    url = "https://nominatim.openstreetmap.org/search"
    try:
        r = requests.get(url, params={"q": address, "format": "json"}, headers={"User-Agent": "salmonshield"})
        results = r.json()
        if not results:
            return None  # address not found
        result = results[0]  # take the best match
        return float(result["lat"]), float(result["lon"])
    except Exception:
        return None  # fail gracefully on network/parse errors



# --- load GeoJSON data files into GeoDataFrames ---
def load_geo(path, crs="EPSG:4326"):
    # return empty dataframe if file missing or empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    try:
        return gpd.read_file(path)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

# load all datasets at startup (runs once when server starts)
streams = load_geo("data/salmon_streams.geojson")         # salmon stream lines
stormwater = load_geo("data/storm_discharge.geojson")     # stormwater drain points
watersheds = load_geo("data/watershed_boundaries.geojson") # watershed boundary polygons
water_quality = load_geo("data/water_quality_pollutants.geojson")  # water quality by census tract
heavy_traffic = load_geo("data/proximity_heavy_traffic_roadways.geojson")  # traffic exposure data

# --- routes ---

# home page
@app.route("/")
def home():
    return render_template("index.html")

# geocode endpoint — converts address text to lat/lon JSON
@app.route("/geocode", methods=["POST"])
def geocode_address():
    address = request.form.get("address")
    if not address:
        return jsonify({"error": "No address provided"}), 400
    coords = geocode(address)
    if not coords:
        return jsonify({"error": "Could not find that address"}), 404
    return jsonify({"lat": coords[0], "lon": coords[1]})

# --- risk scoring helpers ---

# map overall score to a color: 60+ = red, 30+ = yellow, else green
def score_to_color(score):
    if score >= 60:
        return "red"
    elif score >= 30:
        return "yellow"
    else:
        return "green"

# per-stream risk color based on how close the stream is + nearby drain count
def get_risk_color(distance_m, drain_count):
    proximity = 70 * max(0, 1 - distance_m / 1000)  # closer = higher
    drain_boost = min(30, drain_count * 6)           # more drains = higher
    score = max(0, min(100, int(proximity + drain_boost)))
    return score_to_color(score)

# convert a GeoDataFrame to a list of GeoJSON features (handles date columns)
def _geodf_to_features(gdf):
    for col in gdf.select_dtypes(include=["datetime", "datetimetz"]).columns:
        gdf[col] = gdf[col].astype(str)  # dates -> strings for JSON
    return json.loads(gdf.to_json())["features"]

# --- main analysis function ---
def run_analysis(lat, lon):
    user_point = Point(lon, lat)  # shapely point (lon first, lat second)

    # reproject to meters (EPSG:3857) for distance calculations
    user_point_m = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    # early return if no data loaded
    if streams.empty and stormwater.empty:
        return {"nearby_streams": [], "nearby_stormwater": [], "impact_score": 0, "risk_color": "green"}

    # reproject datasets to meters for distance math
    streams_m = streams.to_crs(epsg=3857) if not streams.empty else streams
    stormwater_m = stormwater.to_crs(epsg=3857) if not stormwater.empty else stormwater

    # filter to features within 1 km of user
    nearby_streams = streams_m[streams_m.geometry.distance(user_point_m) < 1000] if not streams_m.empty else streams_m
    nearby_stormwater = stormwater_m[stormwater_m.geometry.distance(user_point_m) < 1000] if not stormwater_m.empty else stormwater_m

    nearest_distance = nearby_streams.geometry.distance(user_point_m).min() if not nearby_streams.empty else 1000

    # --- impact score (0-100) from 3 components ---
    proximity_score = 40 * max(0, 1 - nearest_distance / 1000)  # max 40 pts
    density_score = min(30, len(nearby_streams) * 5)             # max 30 pts
    stormwater_score = min(30, len(nearby_stormwater) * 5)       # max 30 pts
    impact_score = max(0, min(100, int(proximity_score + density_score + stormwater_score)))

    # assign a risk color to each stream based on distance + drain count
    nearby_streams = nearby_streams.copy()
    drain_count = len(nearby_stormwater)
    nearby_streams['riskColor'] = [
        get_risk_color(geom.distance(user_point_m), drain_count)
        for geom in nearby_streams.geometry
    ]

    # --- fallback: find nearest stream anywhere if none within 1 km ---
    nearest_stream_point = None
    nearest_stream_dist_km = None
    nearest_stream_feature = None
    if nearby_streams.empty and not streams_m.empty:
        all_distances = streams_m.geometry.distance(user_point_m)
        closest_idx = all_distances.idxmin()  # index of closest stream
        nearest_stream_dist_km = round(all_distances[closest_idx] / 1000, 1)
        # snap to closest point on the stream line
        closest_geom = streams_m.geometry[closest_idx]
        nearest_pt_m = closest_geom.interpolate(closest_geom.project(user_point_m))
        nearest_pt_4326 = gpd.GeoSeries([nearest_pt_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        nearest_stream_point = [nearest_pt_4326.y, nearest_pt_4326.x]  # [lat, lon]
        # get the full stream geometry for drawing on the map
        row = streams.loc[closest_idx]
        # look up which watershed this stream is in
        ws_name = None
        if not watersheds.empty:
            rep_pt = row.geometry.representative_point()  # a point guaranteed inside the geometry
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

    # convert back to lat/lon (EPSG:4326) for the frontend map
    nearby_streams_4326 = nearby_streams.to_crs(epsg=4326) if not nearby_streams.empty else nearby_streams
    raw_features = nearby_streams_4326.__geo_interface__['features'] if not nearby_streams_4326.empty else []

    # spatial join: match each stream to its watershed name
    if not nearby_streams_4326.empty and not watersheds.empty:
        pts = nearby_streams_4326.copy()
        pts["_rep"] = pts.geometry.representative_point()
        pts = pts.set_geometry("_rep")
        joined = gpd.sjoin(pts, watersheds[["Name", "geometry"]], how="left", predicate="within")
        # map LLID -> watershed name (first match)
        ws_lookup = dict(zip(joined["LLID"], joined["Name"]))
    else:
        ws_lookup = {}

    # deduplicate streams by LLID — keep longest geometry, merge species
    seen = {}
    merged_features = []
    for f in raw_features:
        llid = f["properties"].get("LLID", "")
        sp = f["properties"].get("SPECIES", "")
        coord_count = len(f["geometry"]["coordinates"])

        if llid in seen:
            # merge species into existing entry
            if sp and sp not in seen[llid]["properties"]["allSpecies"]:
                seen[llid]["properties"]["allSpecies"].append(sp)
            # keep the geometry with more coordinates (longer line)
            if coord_count > len(seen[llid]["geometry"]["coordinates"]):
                seen[llid]["geometry"] = f["geometry"]
        else:
            props = dict(f["properties"])
            props["allSpecies"] = [sp] if sp else []
            props["watershed_name"] = ws_lookup.get(llid, None)  # attach watershed
            f["properties"] = props
            seen[llid] = f
            merged_features.append(f)

    # --- nearby watersheds (for map polygons) ---
    nearby_watersheds = []
    if not watersheds.empty:
        buf = 0.03  # ~3km bounding box in degrees
        bbox = box(lon - buf, lat - buf, lon + buf, lat + buf)
        hits = watersheds[watersheds.intersects(bbox)]
        if not hits.empty:
            hits_simple = hits.copy()
            hits_simple["geometry"] = hits_simple.geometry.simplify(0.0005)  # reduce detail for speed
            cols = [c for c in ["Name", "HUC12", "AreaSqKm", "geometry"] if c in hits_simple.columns]
            nearby_watersheds = json.loads(hits_simple[cols].to_json(default=str))["features"]

    # --- water quality lookup (~500m radius) ---
    wq_data = None
    if not water_quality.empty:
        buf_deg = 0.0045
        bbox = box(lon - buf_deg, lat - buf_deg, lon + buf_deg, lat + buf_deg)
        hits = water_quality[water_quality.intersects(bbox)]
        if not hits.empty:
            containing = hits[hits.contains(user_point)]  # prefer exact tract
            tracts = containing if not containing.empty else hits
            total_impairments = int(tracts["TotalUniqueImpairments"].sum())
            max_rank = int(tracts["Water_Quality_Rank"].max())
            wq_data = {
                "total_impairments": total_impairments,
                "max_rank": max_rank,
                "tract_count": len(tracts)
            }

    # --- heavy traffic lookup (~1km radius) ---
    traffic_data = None
    if not heavy_traffic.empty:
        buf_deg = 0.009
        bbox = box(lon - buf_deg, lat - buf_deg, lon + buf_deg, lat + buf_deg)
        hits = heavy_traffic[heavy_traffic.intersects(bbox)]
        if not hits.empty:
            containing = hits[hits.contains(user_point)]  # prefer exact tract
            tracts = containing if not containing.empty else hits
            traffic_data = {
                "ehd_rank": int(tracts["EHD_Rank"].max()),
                "env_exp_rank": int(tracts["Env_Exp_Rank"].max()),
            }

    # pack everything into a dict for the template
    return {
        "nearby_streams": merged_features,
        "nearby_stormwater": _geodf_to_features(nearby_stormwater.to_crs(epsg=4326)) if not nearby_stormwater.empty else [],
        "nearby_watersheds": nearby_watersheds,
        "impact_score": impact_score,
        "risk_color": score_to_color(impact_score),
        "nearest_stream_point": nearest_stream_point,       # [lat, lon] or None
        "nearest_stream_dist_km": nearest_stream_dist_km,   # float or None
        "nearest_stream_feature": nearest_stream_feature,    # GeoJSON feature or None
        "water_quality": wq_data,
        "traffic": traffic_data
    }


# --- GPT-4o suggestions ---
# lazy init so .env loads before we create the client
_openai_client = None
def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client

# build a prompt with real data and ask GPT for actionable suggestions
def get_suggestions(impact_score, discharge_count, road_density, water_quality, precip_forecast):
    prompt = (
        f"You are a salmon habitat conservation advisor. Given this data:\n"
        f"- Danger score: {impact_score}/100\n"
        f"- Stormwater discharge points nearby: {discharge_count}\n"
        f"- Road density: {road_density}\n"
        f"- Water quality: {water_quality}\n"
        f"- Precipitation forecast: {precip_forecast}\n\n"
        "Give exactly two sets of bullet-point suggestions. Each suggestion MUST include "
        "specific numbers, percentages, or measurable actions based on the data above. "
        "For example: 'Reducing impervious surface by 15% in this area would cut runoff volume by ~20%.' "
        "Do NOT write generic advice. Reference the actual numbers provided.\n\n"
        "1. COMMUNITY: 2-3 bullet points a regular person can act on, with concrete numbers.\n"
        "2. SCIENTIST: 2-3 bullet points for a conservation scientist, with technical specifics.\n\n"
        "Return JSON with keys \"community\" and \"scientist\". Each value is a string "
        "with bullet points separated by newlines, each starting with •"
    )
    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # forces valid JSON output
            temperature=0.7,
            max_tokens=300,
        )
        return json.loads(resp.choices[0].message.content)  # parse JSON response
    except Exception as e:
        print(f"GPT suggestion error: {e}")
        return {"community": None, "scientist": None}  # graceful fallback


# --- JSON API endpoint (for programmatic access) ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        lat = float(request.form["latitude"])
        lon = float(request.form["longitude"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid or missing coordinates"}), 400

    data = run_analysis(lat, lon)
    data["suggestions"] = get_suggestions(  # append GPT suggestions
        impact_score=data["impact_score"],
        discharge_count=len(data["nearby_stormwater"]),
        road_density=data.get("traffic"),
        water_quality=data.get("water_quality"),
        precip_forecast="see legend",
    )
    return jsonify(data)


# --- results page (the main user-facing page) ---
@app.route("/results")
def results():
    try:
        lat = float(request.args.get("lat"))  # from URL ?lat=...&lon=...
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return "Missing or invalid lat/lon", 400

    data = run_analysis(lat, lon)       # run spatial analysis
    data["suggestions"] = get_suggestions(  # get GPT suggestions
        impact_score=data["impact_score"],
        discharge_count=len(data["nearby_stormwater"]),
        road_density=data.get("traffic"),
        water_quality=data.get("water_quality"),
        precip_forecast="see legend",
    )

    # pass all data to the Jinja2 template as JSON strings
    return render_template("results.html",
        lat=lat, lon=lon,
        streams_json=json.dumps(data["nearby_streams"]),
        stormwater_json=json.dumps(data["nearby_stormwater"]),
        watersheds_json=json.dumps(data["nearby_watersheds"]),
        impact_score=data["impact_score"],
        risk_color=data["risk_color"],
        nearest_stream_point=json.dumps(data["nearest_stream_point"]),
        nearest_stream_feature_json=json.dumps(data["nearest_stream_feature"]),
        nearest_stream_dist_km=data["nearest_stream_dist_km"],
        water_quality_json=json.dumps(data["water_quality"]),
        traffic_json=json.dumps(data["traffic"]),
        suggestions=data.get("suggestions", {})
    )

# --- NLCD impervious surface tile endpoint ---
# source GeoTIFF — 2024 fractional impervious surface data (Albers projection)
NLCD_TIFF = "data/NLCD_b812e669-a6c9-44dd-a6b7-127872e98e5c/Annual_NLCD_FctImp_2024_CU_C1V1_b812e669-a6c9-44dd-a6b7-127872e98e5c.tiff"

# convert impervious % values (0-100) to colored RGBA pixels
def impervious_to_rgba(band):
    h, w = band.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = band > 0  # skip fully pervious (natural) areas
    pct = band[mask].astype(float) / 100.0
    # color gradient: green (low%) -> yellow (mid%) -> red (high%)
    rgba[mask, 0] = (pct * 255).astype(np.uint8)           # red channel up
    rgba[mask, 1] = ((1 - pct) * 200).astype(np.uint8)     # green channel down
    rgba[mask, 2] = 30                                       # blue constant
    rgba[mask, 3] = (60 + pct * 140).astype(np.uint8)       # more opaque at high %
    return rgba

# serves a PNG image of impervious surface for the current map view
@app.route("/nlcd_tile")
def nlcd_tile():
    try:
        # bounding box from the frontend (in lat/lon degrees)
        west = float(request.args["west"])
        south = float(request.args["south"])
        east = float(request.args["east"])
        north = float(request.args["north"])
        width = int(request.args.get("width", 512))   # output image width
        height = int(request.args.get("height", 512))  # output image height
    except (KeyError, ValueError):
        return "Missing bbox params", 400

    # reproject bbox to Web Mercator (EPSG:3857) to match Leaflet tiles
    merc_bounds = transform_bounds("EPSG:4326", "EPSG:3857", west, south, east, north)
    dst_transform = from_bounds(*merc_bounds, width, height)  # pixel-to-coordinate mapping
    dst_array = np.zeros((height, width), dtype=np.uint8)     # output raster

    with rasterio.open(NLCD_TIFF) as src:
        # warp from source CRS (Albers) to 3857 so it lines up on the map
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.bilinear,
        )

    # colorize and send as a transparent PNG
    rgba = impervious_to_rgba(dst_array)
    img = Image.fromarray(rgba, "RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# run with: python app.py (starts on http://localhost:5000)
if __name__ == "__main__":
    app.run(debug=True, port=5000)
