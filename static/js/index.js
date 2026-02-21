// This example requires the Places library. Include the libraries=places
// parameter when you first load the API. For example:
// <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=places">
function initMap() {
  const map = new google.maps.Map(document.getElementById("map"), {
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

    map.setCenter(place.location);
    map.setZoom(17);

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
