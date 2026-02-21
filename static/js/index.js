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
  const useLocationBtn = document.getElementById("use-location");
  const readout = document.getElementById("location-readout");

  function showLocation(lat, lng, label = "Your location") {
    // redirect to results page with coordinates
    window.location.href = `/results?lat=${lat}&lon=${lng}`;
  }

  if (useLocationBtn) {
    useLocationBtn.addEventListener("click", () => {
      if (!navigator.geolocation) {
        if (readout) readout.textContent = "Geolocation is not supported in this browser.";
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          showLocation(latitude, longitude, "Your current location");
        },
        (err) => {
          if (readout) readout.textContent = `Location error: ${err.message}`;
        },
        { enableHighAccuracy: true, timeout: 10000 }
      );
    });
  }

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

    showLocation(place.location.lat(), place.location.lng(), place.displayName || "Selected place");
  });
}

window.initMap = initMap;
