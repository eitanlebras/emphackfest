# import statements
from flask import Flask, render_template, request, jsonify
import geopandas as gpd
from shapely.geometry import Point

app = Flask(__name__)

# load geoJSON datasets
streams = gpd.read_file("data/salmon_streams.geojson")
stormwater = gpd.read_file("data/stormwater_discharge.geojson")

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
    streams_m = streams.to_crs(epsg=3857)
    stormwater_m = stormwater.to_crs(epsg=3857)
    user_point_m = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    # nearby features within 1 km
    # filters out streams and stormwater drains that are within 1 km of the user's location using boolean indexing
    # if under 1000 meters = true, otherwise false
    nearby_streams = streams_m[streams_m.geometry.distance(user_point_m) < 1000] # within 1 km
    nearby_stormwater = stormwater_m[stormwater_m.geometry.distance(user_point_m) < 1000]

    # calculating impact score based on proximity and number of features
    nearest_distance = nearby_streams.geometry.distance(user_point_m).min() if not nearby_streams.empty else 1000 # if it can't find any nearby streams, it defaults to 1km away
    impact_score = max(0, min(100, int((1 / (nearest_distance/100 + 1)) * 50 + len(nearby_stormwater) * 10))) # score calculation so that the max is 100 and that the min is 0; the closer the nearest stream and the more stormwater drains, the higher the score
    
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
        lambda row: get_risk_color(row.geometry.distance(user_point_m), len(nearby_stormwater)), # in each row, calculates distance and number of storm drains
        axis=1 # goes row by row in the dataset
    )

    # prepare json results for the frontend
    results = {
        "nearby_streams": nearby_streams.__geo_interface__['features'],
        "nearby_stormwater": nearby_stormwater.to_dict(orient="records"),
        "impact_score": impact_score,
        "risk_color": get_risk_color(nearest_distance, len(nearby_stormwater))
    }

    # sends python dictionary as json response to the frontend
    return jsonify(results)