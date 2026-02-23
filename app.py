# --- setup & imports ---
import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # allow reading very large GeoJSON files
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))  # load API keys from .env
from flask import Flask, render_template, request, jsonify, send_file
import geopandas as gpd        # geospatial dataframes (like pandas but for maps)
from shapely.geometry import Point, box  # geometric shapes for spatial queries
import json
import math
from concurrent.futures import ThreadPoolExecutor
import requests
import openai
import pandas as pd
import rasterio
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
    if not os.path.exists(path) or os.path.getsize(path) == 0: # if file doesn't exist or is empty, return empty GeoDataFrame
        return gpd.GeoDataFrame(geometry=[], crs=crs) # return empty GeoDataFrame with specified CRS
    try: 
        return gpd.read_file(path)
    except Exception as e: # exception handling for file read errors (e.g. invalid GeoJSON)
        print(f"Warning: failed to read {path}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

# load and aggregate spawning ground surveys by stream ID
def load_spawning_surveys(path):
    """Load WDFW spawning ground surveys and aggregate by LLID.
    Returns a dict: {llid_int: {species, latest_year, total_redds, survey_count}}
    """
    if not os.path.exists(path): # if file doesn't exist, return empty dict
        return {}
    try:
        df = pd.read_csv(path, low_memory=False, # only load relevant columns to save memory
                         usecols=["LLID", "Species", "Run Year", "Total Visible Redds"])
        df = df.dropna(subset=["LLID"]) # drop rows without LLID since we can't associate them with a stream
        df["LLID"] = df["LLID"].astype("int64") # ensure LLID is integer for consistent dict keys

        # unique species observed per stream across all years
        species_by_llid = ( # group by LLID, collect unique non-null species into a sorted list
            df.groupby("LLID")["Species"] 
            .apply(lambda x: sorted(set(x.dropna()))) # drop null species, get unique values, sort alphabetically
            .to_dict()
        )

        # most recent year's total visible redds per stream
        df_r = df.dropna(subset=["Run Year", "Total Visible Redds"]).copy() # only consider rows with valid year and redd count for this part
        df_r["Run Year"] = df_r["Run Year"].astype(int) # ensure year is integer for proper max calculation
        yearly = df_r.groupby(["LLID", "Run Year"])["Total Visible Redds"].sum().reset_index() # sum redds per LLID per year (in case of multiple surveys in the same year)
        latest_idx = yearly.groupby("LLID")["Run Year"].idxmax() # get index of the row with the latest year for each LLID
        latest = yearly.loc[latest_idx].set_index("LLID") # create a dataframe indexed by LLID with columns for latest year and redd count

        # survey count per stream
        survey_count = df.groupby("LLID").size().to_dict()

        result = {} # creates dict with LLID as key and aggregated survey data as value
        for llid in df["LLID"].unique(): # iterate over each unique LLID to build the result dict
            entry = { # initialize entry with species list and placeholders for year/redds
                "species": species_by_llid.get(llid, []),
                "latest_year": None,
                "total_redds": None,
                "survey_count": survey_count.get(llid, 0),
            }
            if llid in latest.index: # if we have valid year/redd data for this LLID, fill in those fields
                row = latest.loc[llid]
                entry["latest_year"] = int(row["Run Year"])
                entry["total_redds"] = int(row["Total Visible Redds"])
            result[int(llid)] = entry
        print(f"Loaded spawning surveys for {len(result)} streams.")
        return result
    except Exception as e:
        print(f"Warning: failed to load spawning surveys: {e}")
        return {}

# loads all GeoJSON files at startup
streams = load_geo("data/salmon_streams.geojson")
stormwater = load_geo("data/storm_discharge.geojson")
watersheds = load_geo("data/watershed_boundaries.geojson")
water_quality = load_geo("data/water_quality_pollutants.geojson")
heavy_traffic = load_geo("data/proximity_heavy_traffic_roadways.geojson")
spawning_surveys = load_spawning_surveys("data/WDFW-Spawning_Ground_Surveys.csv")

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
    proximity = 70 * max(0, 1 - distance_m / 1000)  # closer = higher danger
    drain_boost = min(30, drain_count * 6)           # more drains = higher danger
    score = max(0, min(100, int(proximity + drain_boost)))
    return score_to_color(score)

# convert a GeoDataFrame to a list of GeoJSON features (handles date columns)
def _geodf_to_features(gdf):
    for col in gdf.select_dtypes(include=["datetime", "datetimetz"]).columns:
        gdf[col] = gdf[col].astype(str)  # converts dates -> strings for JSON
    return json.loads(gdf.to_json())["features"] # extract list of features from the full GeoJSON dict

# --- server-side data fetchers for scoring ---

# haversine formula (calculate distance between 2 points) (server-side version for road density calc)
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1) # convert lat/lon differences to radians
    dlon = math.radians(lon2 - lon1) # haversine formula to calculate great-circle distance between two points on the Earth
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# fetch road density (km/km²) from Overpass API within 500m
def fetch_road_density(lat, lon, radius=500):
    area_km2 = math.pi * (radius / 1000) ** 2
    query = f'[out:json][timeout:15];(way["highway"](around:{radius},{lat},{lon}););out geom;' # query to fetch all highways (roads) within the specified radius of the given lat/lon point, including their geometry for length calculation
    # calculate total length of roads in the area and divide by area to get density; handle errors gracefully
    try:
        r = requests.post('https://overpass-api.de/api/interpreter', # send the query to the Overpass API to get road data
                          data={'data': query}, timeout=20)
        data = r.json()
        elements = data.get('elements', []) # extract the list of road elements from the API response; if no elements, return None for density
        if not elements:
            return None
        total_m = 0
        for way in elements: # iterate over each road element, calculate its length using the geometry points, and sum up the total length of roads in meters
            geom = way.get('geometry', [])
            for i in range(len(geom) - 1):
                total_m += _haversine(geom[i]['lat'], geom[i]['lon'],
                                      geom[i + 1]['lat'], geom[i + 1]['lon'])
        return round(total_m / 1000 / area_km2, 1)
    except Exception:
        return None

# fetch 3-day precipitation forecast from Open-Meteo (free, no key)
def fetch_precipitation(lat, lon):
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': lat, 'longitude': lon,
            'daily': 'precipitation_sum,precipitation_probability_max',
            'precipitation_unit': 'inch',
            'timezone': 'auto',
            'forecast_days': 3
        }, timeout=10)
        data = r.json()
        daily = data['daily']
        return {
            'days': daily['time'],
            'precip_inches': daily['precipitation_sum'],
            'precip_prob': daily['precipitation_probability_max'],
            'total_inches': round(sum(p or 0 for p in daily['precipitation_sum']), 2)
        }
    except Exception:
        return None

# sample NLCD impervious surface % at a single point
def sample_impervious(lat, lon):
    # open the GeoTIFF, transform the input lat/lon to the raster's CRS, sample the pixel value, and return it as an integer percentage (0-100); handle errors gracefully and treat values >100 as nodata
    try:
        with rasterio.open(NLCD_TIFF) as src:
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
            vals = list(src.sample([(xs[0], ys[0])])) # sample the raster value at the transformed coordinates; returns a list of arrays, take the first one and convert to int
            val = int(vals[0][0])
            return val if 0 <= val <= 100 else 0  # values >100 are NLCD nodata/fill
    except Exception:
        return 0

# --- weighted danger score (0-100) from 7 normalized features ---
def calculate_score(distance_m, discharge_count, road_density, ehd_rank,
                    impervious_pct, precip_inches, water_quality_score):
    # normalize each feature to 0-1
    n_distance = max(0.0, 1 - distance_m / 1000)              # closer = higher risk
    n_discharge = min(1.0, discharge_count / 8)                # cap at 8 drains
    n_road = min(1.0, (road_density or 0) / 25)               # cap at 25 km/km²
    n_traffic = (ehd_rank or 0) / 10                           # 1-10 scale
    n_imperv = (impervious_pct or 0) / 100                     # 0-100%
    n_rain = min(1.0, (precip_inches or 0) / 1.5)             # cap at 1.5 in
    # water quality 0-100 (higher=better), invert so bad water = high risk
    n_wq = 1 - ((water_quality_score if water_quality_score is not None else 50) / 100)

    # weighted sum (weights sum to 1.0)
    score = (
        0.22 * n_distance +
        0.15 * n_discharge +
        0.18 * n_road +
        0.12 * n_traffic +
        0.13 * n_imperv +
        0.12 * n_rain +
        0.08 * n_wq
    )
    return max(0, min(100, int(round(score * 100))))


# --- main analysis function ---
def run_analysis(lat, lon):
    user_point = Point(lon, lat)  # turns address shapely point (lon first, lat second)

    # reproject to meters (EPSG:3857) for distance calculations
    user_point_m = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    # early return if no data loaded
    if streams.empty and stormwater.empty:
        return {
            "nearby_streams": [], "nearby_stormwater": [], "nearby_watersheds": [],
            "impact_score": 0, "risk_color": "green",
            "nearest_stream_point": None, "nearest_stream_dist_km": None,
            "nearest_stream_feature": None,
            "water_quality": None, "traffic": None,
            "road_density": None, "precipitation": None, "impervious_pct": 0,
        }

    # start async fetches in background while we do spatial analysis
    pool = ThreadPoolExecutor(max_workers=3) # speeds up the page load by fetching road density, precipitation, and impervious surface data in parallel with the spatial queries
    rd_future = pool.submit(fetch_road_density, lat, lon)
    precip_future = pool.submit(fetch_precipitation, lat, lon)
    imperv_future = pool.submit(sample_impervious, lat, lon)

    # reproject datasets to meters (EPSG:3857) for distance math
    streams_m = streams.to_crs(epsg=3857) if not streams.empty else streams 
    stormwater_m = stormwater.to_crs(epsg=3857) if not stormwater.empty else stormwater

    # filter to features within 1 km of user, keeps empty dataframes if no features or if original was empty
    nearby_streams = streams_m[streams_m.geometry.distance(user_point_m) < 1000] if not streams_m.empty else streams_m
    nearby_stormwater = stormwater_m[stormwater_m.geometry.distance(user_point_m) < 1000] if not stormwater_m.empty else stormwater_m

    nearest_distance = nearby_streams.geometry.distance(user_point_m).min() if not nearby_streams.empty else 1000 # if no nearby streams, treat as 1 km for scoring purposes

    # assign a risk color to each stream based on distance + drain count
    nearby_streams = nearby_streams.copy() # avoid SettingWithCopyWarning when adding new column
    drain_count = len(nearby_stormwater)
    nearby_streams['riskColor'] = [
        get_risk_color(geom.distance(user_point_m), drain_count) # calls get_risk_color for each stream geometry to determine its risk color based on distance and nearby drain count
        for geom in nearby_streams.geometry
    ]

    # --- fallback: find nearest stream anywhere if none within 1 km ---
    nearest_stream_point = None
    nearest_stream_dist_km = None
    nearest_stream_feature = None
    if nearby_streams.empty and not streams_m.empty: # if no streams within 1 km but we have stream data, find the closest stream overall to provide some context to the user instead of showing an empty map
        all_distances = streams_m.geometry.distance(user_point_m) # calculate distance from user point to all streams in meters
        closest_idx = all_distances.idxmin()  # index of closest stream
        nearest_stream_dist_km = round(all_distances[closest_idx] / 1000, 1)
        # snap to closest point on the stream line
        closest_geom = streams_m.geometry[closest_idx]
        nearest_pt_m = closest_geom.interpolate(closest_geom.project(user_point_m)) # finds the point on the stream geometry that is closest to the user point by projecting the user point onto the stream line and interpolating to get the exact coordinates of that closest point in meters
        nearest_pt_4326 = gpd.GeoSeries([nearest_pt_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0] # convert the nearest point back to lat/lon for frontend display; this is the point we will show on the map as the "nearest stream location" even if it's more than 1 km away, so users can see where the closest stream actually is in relation to their location
        nearest_stream_point = [nearest_pt_4326.y, nearest_pt_4326.x]
        # get the full stream geometry for drawing on the map
        row = streams.loc[closest_idx]
        # look up which watershed this stream is in
        ws_name = None
        # if we have watershed data, find the one that is inside the watershed polygon in the dataset
        if not watersheds.empty:
            rep_pt = row.geometry.representative_point()  # a point guaranteed inside the geometry
            containing = watersheds[watersheds.contains(rep_pt)]
            # if multiple watersheds contain the point (e.g. due to data issues), just take the first one; if none contain it, ws_name stays None
            if not containing.empty:
                ws_name = containing.iloc[0].get("Name", None)
        nearest_llid = row.get("LLID") # get the stream's LLID to look up spawning survey data; if LLID is missing or invalid, we'll just skip the survey info for this stream
        try:
            nearest_survey = spawning_surveys.get(int(nearest_llid), {}) if nearest_llid else {} # look up spawning survey data for this stream using its LLID; if LLID is missing or not an integer, treat as no survey data
        except (ValueError, TypeError):
            nearest_survey = {}
        nearest_stream_feature = { # construct a GeoJSON feature for the nearest stream to show on the map, including properties for the name, risk color, watershed, and spawning survey data if available
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "LLID_STRM_NAME": row.get("LLID_STRM_NAME", "Nearest stream"),
                "riskColor": get_risk_color(all_distances[closest_idx], drain_count),
                "watershed_name": ws_name,
                "survey_species": nearest_survey.get("species", []),
                "latest_redd_year": nearest_survey.get("latest_year"),
                "total_redds": nearest_survey.get("total_redds"),
                "survey_count": nearest_survey.get("survey_count", 0),
            }
        }

    # convert back to lat/lon (EPSG:4326) for the frontend map
    nearby_streams_4326 = nearby_streams.to_crs(epsg=4326) if not nearby_streams.empty else nearby_streams
    raw_features = nearby_streams_4326.__geo_interface__['features'] if not nearby_streams_4326.empty else []

    # spatial join: match each stream to its watershed name
    if not nearby_streams_4326.empty and not watersheds.empty:
        pts = nearby_streams_4326.copy() # avoid modifying original GeoDataFrame by making a copy
        pts["_rep"] = pts.geometry.representative_point()
        pts = pts.set_geometry("_rep") # use the representative point for the spatial join to ensure we get a single watershed match per stream
        joined = gpd.sjoin(pts, watersheds[["Name", "geometry"]], how="left", predicate="within") # perform spatial join to find which watershed polygon each stream's representative point falls within; this allows us to attach the watershed name to each stream feature for display on the frontend
        # map LLID -> watershed name (first match)
        ws_lookup = dict(zip(joined["LLID"], joined["Name"]))
    else:
        ws_lookup = {}

    # deduplicate streams by LLID — keep longest geometry, merge species
    seen = {}
    merged_features = []
    for f in raw_features: # iterate over each stream feature from the nearby streams GeoDataFrame
        llid = f["properties"].get("LLID", "")
        sp = f["properties"].get("SPECIES", "")
        coord_count = len(f["geometry"]["coordinates"])

        if llid in seen: # if we've already seen a stream with this LLID, we want to merge the species lists and keep the geometry with more coordinates
            # merge species into existing entry
            if sp and sp not in seen[llid]["properties"]["allSpecies"]: # if this feature has a species and it's not already in the list for this LLID, add it to the list of all species observed for this stream
                seen[llid]["properties"]["allSpecies"].append(sp)
            # keep the geometry with more coordinates (longer line)
            if coord_count > len(seen[llid]["geometry"]["coordinates"]): # if this feature has more coordinates, than replace the previous feature
                seen[llid]["geometry"] = f["geometry"]
        else: # first time seeing this LLID, add it to the seen dict and initialize allSpecies list
            props = dict(f["properties"]) # 
            props["allSpecies"] = [sp] if sp else []
            # attach watershed name from spatial join
            props["watershed_name"] = ws_lookup.get(llid, None)
            # attach spawning survey data keyed by LLID
            try:
                survey = spawning_surveys.get(int(llid), {}) # look up spawning survey data for this stream using its LLID; if LLID is missing or not an integer, treat as no survey data
            except (ValueError, TypeError):
                survey = {}
            # add survey data to properties if available; this allows us to show users which salmon species have been observed in surveys of this stream, how many redds were counted in the most recent survey, and how many total surveys have been conducted
            props["survey_species"] = survey.get("species", [])
            props["latest_redd_year"] = survey.get("latest_year")
            props["total_redds"] = survey.get("total_redds")
            props["survey_count"] = survey.get("survey_count", 0)
            f["properties"] = props
            seen[llid] = f
            merged_features.append(f)

    # --- nearby watersheds (for map polygons) ---
    nearby_watersheds = []
    if not watersheds.empty:
        buf = 0.03  # ~3km bounding box in degrees
        bbox = box(lon - buf, lat - buf, lon + buf, lat + buf) # create a bounding box around the user's location to quickly filter the watershed polygons to only those that are potentially nearby
        hits = watersheds[watersheds.intersects(bbox)] # filter the watershed polygons to only those that intersect with the bounding box
        if not hits.empty: # simply rendering of the polygons to load faster on the front end
            hits_simple = hits.copy()
            hits_simple["geometry"] = hits_simple.geometry.simplify(0.0005)  # reduce detail for speed
            cols = [c for c in ["Name", "HUC12", "AreaSqKm", "geometry"] if c in hits_simple.columns]
            nearby_watersheds = json.loads(hits_simple[cols].to_json(default=str))["features"] # convert the nearby watershed GeoDataFrame to a list of GeoJSON features

    # --- water quality lookup (~500m radius) ---
    wq_data = None
    if not water_quality.empty:
        buf_deg = 0.0045 # ~500m in degrees
        bbox = box(lon - buf_deg, lat - buf_deg, lon + buf_deg, lat + buf_deg) # create a bounding box around the user's location to filter the water quality polygons to only those that are potentially nearby
        hits = water_quality[water_quality.intersects(bbox)] # filter the water quality polygons to only those that intersects with the boundary box of the user's address
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
        buf_deg = 0.009 # ~1km in degrees
        bbox = box(lon - buf_deg, lat - buf_deg, lon + buf_deg, lat + buf_deg) # same as previous lookups
        hits = heavy_traffic[heavy_traffic.intersects(bbox)]
        if not hits.empty:
            containing = hits[hits.contains(user_point)]  # prefer exact tract
            tracts = containing if not containing.empty else hits
            traffic_data = {
                "ehd_rank": int(tracts["EHD_Rank"].max()),
                "env_exp_rank": int(tracts["Env_Exp_Rank"].max()),
            }

    # --- collect async results and compute weighted danger score ---
    road_density_val = rd_future.result()
    precip_data = precip_future.result()
    impervious_pct = imperv_future.result()
    pool.shutdown(wait=False)

    # extract values for the scoring function
    ehd_rank = traffic_data["ehd_rank"] if traffic_data else None
    precip_total = precip_data["total_inches"] if precip_data else None
    # convert rank 1-10 (higher=worse) to quality 0-100 (higher=better)
    wq_score = round((10 - wq_data["max_rank"]) / 9 * 100) if wq_data else None

    # calculate overall impact score from all features
    impact_score = calculate_score(
        distance_m=nearest_distance,
        discharge_count=drain_count,
        road_density=road_density_val,
        ehd_rank=ehd_rank,
        impervious_pct=impervious_pct,
        precip_inches=precip_total,
        water_quality_score=wq_score,
    )

    # pack everything into a dict for the template
    return {
        "nearby_streams": merged_features,
        "nearby_stormwater": _geodf_to_features(nearby_stormwater.to_crs(epsg=4326)) if not nearby_stormwater.empty else [],
        "nearby_watersheds": nearby_watersheds,
        "impact_score": impact_score,
        "risk_color": score_to_color(impact_score),
        "nearest_stream_point": nearest_stream_point,
        "nearest_stream_dist_km": nearest_stream_dist_km,
        "nearest_stream_feature": nearest_stream_feature,
        "water_quality": wq_data,
        "traffic": traffic_data,
        "road_density": road_density_val,         # km/km² or None
        "precipitation": precip_data,              # dict with days/inches/prob or None
        "impervious_pct": impervious_pct,           # 0-100 integer
    }


# --- GPT-4o suggestions ---
# lazy init so .env loads before we create the client
_openai_client = None
def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client

# build a prompt with real data and ask GPT for actionable suggestions
def get_suggestions(impact_score, discharge_count, road_density, water_quality, precip_forecast):
    prompt = ( # gpt prompt to generate specific suggestions for the user
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

    data = run_analysis(lat, lon) # run the same analysis as the main results page but return all data as JSON instead of rendering a template
    precip = data.get("precipitation") # fetch the precipitation data from the analysis results to include in the suggestions; this allows the GPT suggestions to reference the specific forecasted precipitation amounts
    data["suggestions"] = get_suggestions( # generate GPT suggestions based on the analysis results
        impact_score=data["impact_score"],
        discharge_count=len(data["nearby_stormwater"]),
        road_density=f"{data['road_density']} km/km²" if data.get('road_density') else "unavailable",
        water_quality=data.get("water_quality"),
        precip_forecast=f"{precip['total_inches']} inches over 3 days" if precip else "unavailable",
    )
    return jsonify(data)


# --- results page (the main user-facing page) ---
@app.route("/results")
def results():
    try:
        lat = float(request.args.get("lat"))  # changes to lat/lon
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return "Missing or invalid lat/lon", 400

    data = run_analysis(lat, lon)       # run spatial analysis
    precip = data.get("precipitation")
    data["suggestions"] = get_suggestions( # contrary to /analyze, we generate GPT suggestions here but we will pass them to the template for display on the results page instead of including them in a JSON API response
        impact_score=data["impact_score"],
        discharge_count=len(data["nearby_stormwater"]),
        road_density=f"{data['road_density']} km/km²" if data.get('road_density') else "unavailable",
        water_quality=data.get("water_quality"),
        precip_forecast=f"{precip['total_inches']} inches over 3 days" if precip else "unavailable",
    )

    # pass all data to the Jinja template as JSON strings
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
        road_density_value=data["road_density"],
        precipitation_json=json.dumps(data["precipitation"]),
        impervious_pct=data["impervious_pct"],
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
    rgba[mask, 0] = (pct * 255).astype(np.uint8)           # red increases with impervious %
    rgba[mask, 1] = ((1 - pct) * 200).astype(np.uint8)     # green decreases with impervious %
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
        # warp from source CRS (Albers) to 3857 so it lines up on the map (matches Leaflet's default CRS)
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
