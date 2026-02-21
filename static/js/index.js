// This example requires the Places library. Include the libraries=places
// parameter when you first load the API. For example:
// <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=places">
function initMap() {
  const mapEl = document.getElementById("map");
  const map = new google.maps.Map(mapEl, {
    center: { lat: 40.749933, lng: -73.98633 },
    zoom: 13,
    mapTypeControl: false,
  });

  const pacContainer = document.getElementById("pac-container");
  const autocomplete = new google.maps.places.PlaceAutocompleteElement();
  pacContainer.replaceChildren(autocomplete);

  const infowindow = new google.maps.InfoWindow();
  const marker = new google.maps.Marker({ map });

  autocomplete.addEventListener("gmp-select", async ({ placePrediction }) => {
    if (!placePrediction) {
      return;
    }

    const place = placePrediction.toPlace();
    await place.fetchFields({
      fields: ["location", "formattedAddress", "displayName"],
    });

    if (!place.location) {
      window.alert("No details available for that input.");
      return;
    }

    mapEl.style.display = "block";
    google.maps.event.trigger(map, "resize");
    map.setCenter(place.location);
    map.setZoom(17);

    // store lat and lng as global variables so map.js can access them
    window.userLat = place.location.lat();
    window.userLng = place.location.lng();

    // call function from map.js
    analyzeAddress(window.userLat, window.userLng);

    marker.setPosition(place.location);
    marker.setVisible(true);

    const name = place.displayName || "";
    const address = place.formattedAddress || "";
    infowindow.setContent(
      `<strong>${name}</strong><br>${address}`
    );
    infowindow.open(map, marker);
  });
}

window.initMap = initMap;
