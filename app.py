# import statements
from flask import Flask, render_template, request, jsonify
import geopandas as gpd
from shapely.geometry import Point
import os

app = Flask(__name__)

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

@app.route("/analyze", methods=["POST"])
def analyze():
    # get user input and their address, convert to point geometry with latitude and longitude
    lat = float(request.form["latitude"])
    lon = float(request.form["longitude"])
    user_point = Point(lon, lat)

     # converts distances to meters for more accurate calculations
    streams_m = streams.to_crs(epsg=3857) # epsg:3857 is a common coordinate reference system that uses meters as units, which allows for accurate distance calculations (better than latitude and longitude)
    stormwater_m = stormwater.to_crs(epsg=3857)
    user_point_m = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0] # epsg:4326 is the standard coordinate reference system for latitude and longitude, so we first create a GeoSeries (a series that stores Shapely geometric objects) with the user's point in that CRS and then convert it to epsg:3857 for distance calculations

    # nearby features within 1 km
    # filters out streams and stormwater drains that are within 1 km of the user's location using boolean indexing
    # if the nearby stream is within 1000 meters, the data cell = true, otherwise false
    nearby_streams = streams_m[streams_m.geometry.distance(user_point_m) < 1000] # within 1 km
    nearby_stormwater = stormwater_m[stormwater_m.geometry.distance(user_point_m) < 1000]

    # calculating impact score based on proximity and number of features
    nearest_distance = nearby_streams.geometry.distance(user_point_m).min() if not nearby_streams.empty else 1000 # if it can't find any nearby streams, it defaults to 1km away
    impact_score = max(0, min(100, int((1 / (nearest_distance/100 + 1)) * 50 + len(nearby_stormwater) * 10))) # score calculation so that the max is 100 and that the min is 0; the closer the nearest stream and the more stormwater drains, the higher the score; first takes min of the raw score and 100 to ensure score cannot be over 100, then takes the max of that calculation and 0 to ensure score can't be less than 0
    
    # risk color based on distance and number of stormwater drains
    def get_risk_color(dist, drains):
        if drains >= 5 or dist < 200:
            return "red"
        elif drains >= 2:
            return "yellow"
        else:
            return "green"

    # uploads risk color to the dataset for nearby streams
    nearby_streams['riskColor'] = nearby_streams.apply( # creates new riskColor column in dataset
        lambda row: get_risk_color(row.geometry.distance(user_point_m), len(nearby_stormwater)), # lambda (anonymous) function that uses the get_risk_color function to determine the risk color and then uploads that to the riskColor column for each stream
        axis=1 # goes row by row in the dataset
    )

    # render results page with impact score, risk color, and nearby features
    return render_template(
        "results.html",
        lat=lat,
        lon=lon,
        impact_score=impact_score,
        overall_risk=get_risk_color(nearest_distance, len(nearby_stormwater)),
        nearby_streams=nearby_streams.__geo_interface__['features'],
        nearby_stormwater=nearby_stormwater.to_dict(orient="records")
    )