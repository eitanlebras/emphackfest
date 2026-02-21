# import statements
from flask import Flask, render_template, request, jsonify
import geopandas as gpd
from shapely.geometry import Point
import os
import json
import requests

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
    if score >= 70:
        return "red"
    elif score >= 30:
        return "yellow"
    else:
        return "green"


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

    # impact score: closer streams + more stormwater drains = higher score (0-100)
    impact_score = max(0, min(100, int((1 / (nearest_distance/100 + 1)) * 50 + len(nearby_stormwater) * 10)))

    # add risk color to each stream row
    nearby_streams = nearby_streams.copy()
    drain_count = len(nearby_stormwater)
    nearby_streams['riskColor'] = [
        get_risk_color(geom.distance(user_point_m), drain_count)
        for geom in nearby_streams.geometry
    ]

    # convert streams back to lat/lon for Leaflet (Leaflet uses EPSG:4326)
    nearby_streams_4326 = nearby_streams.to_crs(epsg=4326) if not nearby_streams.empty else nearby_streams

    return {
        "nearby_streams": nearby_streams_4326.__geo_interface__['features'] if not nearby_streams_4326.empty else [],
        "nearby_stormwater": nearby_stormwater.to_crs(epsg=4326).to_dict(orient="records") if not nearby_stormwater.empty else [],
        "impact_score": impact_score,
        "risk_color": score_to_color(impact_score)
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
        impact_score=data["impact_score"],
        risk_color=data["risk_color"]
    )

# allows running the app directly with "python app.py" instead of "flask run"
if __name__ == "__main__":
    app.run(debug=True, port=5000)
