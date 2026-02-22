// Google Places autocomplete — lets users type an address and redirects to results
function initMap() {
  const mapEl = document.getElementById("map");

  // optional preview map on the home page
  let map, marker;
  if (mapEl) {
    map = new google.maps.Map(mapEl, {
      center: { lat: 40.749933, lng: -73.98633 },
      zoom: 13,
      mapTypeControl: false,
    });
    marker = new google.maps.Marker({ map });
  }

  // attach autocomplete widget to the search container
  const pacContainer = document.getElementById("pac-container");
  const autocomplete = new google.maps.places.PlaceAutocompleteElement();
  pacContainer.replaceChildren(autocomplete);

  const infowindow = new google.maps.InfoWindow();
  const useLocationBtn = document.getElementById("use-location");
  const readout = document.getElementById("location-readout");

  // navigate to results page with lat/lon
  function showLocation(lat, lng, label = "Your location") {
    window.location.href = `/results?lat=${lat}&lon=${lng}`;
  }

  // "Use My Location" button — gets GPS coordinates from the browser
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

  // when user picks an address from autocomplete dropdown
  autocomplete.addEventListener("gmp-select", async ({ placePrediction }) => {
    if (!placePrediction) return;

    // fetch the full place details (lat/lon)
    const place = placePrediction.toPlace();
    await place.fetchFields({
      fields: ["location", "formattedAddress", "displayName"],
    });

    if (!place.location) {
      window.alert("No details available for that input.");
      return;
    }

    // redirect to results with the selected coordinates
    const lat = place.location.lat();
    const lng = place.location.lng();
    window.location.href = `/results?lat=${lat}&lon=${lng}`;
  });
}

// make initMap available as a global callback for Google Maps script
window.initMap = initMap;
