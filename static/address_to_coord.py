import requests

def geocode(address):
    # Nominatim is OpenStreetMap's free geocoding API — no token needed
    url = "https://nominatim.openstreetmap.org/search"
    
    # send the address as a query, ask for JSON back
    r = requests.get(url, params={"q": address, "format": "json"}, headers={"User-Agent": "salmonshield"})
    
    # grab the first result (most relevant match)
    result = r.json()[0]
    
    # return as lat, lon floats
    return float(result["lat"]), float(result["lon"])

print(geocode("3528 175th Ave Ne Redmond, WA"))