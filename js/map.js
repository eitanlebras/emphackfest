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