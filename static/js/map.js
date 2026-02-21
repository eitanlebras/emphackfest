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

// Analyze a user inputed address:
async function analyzeAdress(lat, lng) {

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
