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

    // grab lat and lng from the selected place
    const lat = place.location.lat();
    const lng = place.location.lng();

    // send user to the results page with coordinates in the URL
    window.location.href = "/results?lat=" + lat + "&lon=" + lng;
  });
}

window.initMap = initMap;
