// Default map
var map = L.map('map').setView([47.6, -122.3], 12);

// Mapbox tiles
L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token=YOUR_TOKEN', {
  attribution: 'Map data &copy; OpenStreetMap contributors',
  id: 'mapbox/streets-v11',
  tileSize: 512,
  zoomOffset: -1,
  accessToken: 'YOUR_TOKEN'
}).addTo(map);

// Analyze a user inputed address in three step process: 
// 1) Geocode address into lat/lng using OpenStreetMap Nominatim
// 2) Move map + drop marker
// 3) Call Flask backend /analyze to compute nearest stream + score
async function analyzeAdress(address) {
  //Geocoding requestL
  //Nominatim returns a list of possible matches in JSON format
  const geo = await fetch(
    `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(address)}&format=json`
  );
  const geoData = await geo.json();

  // take the first match (assumed best match) and extract coords)
  const lag = parseFloat(geoData[0].lat);
  const lng = parseFloat(geoData[0].lon);

  //Recenter the map and zoom in
  map.setView([lat,lng], 14);

  //Drop a pin at the user location and show a popup
  L.marker([lat, lng]).addTo(map).bindPopup('Your location').openPopup();

  // Call Flash backend
  // The backend uses lat/lng to find nearest stream and calculate scorre
  const res = await fetch(`/analyze?lat=${lat}&lng=${lng}`);
  const data = await res.json();

  // For now, just print the results in the browser console.
  // (In a real UI, you'd show these values in the page.)
  console.log(data);
};
