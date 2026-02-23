# SalmonShield

SalmonShield is a web app that lets you type in your address and see how your neighborhood affects nearby salmon streams through stormwater runoff.

When it rains, water picks up oil, chemicals, and other pollutants as it flows across roads and parking lots. That water goes into storm drains, which often discharge directly into salmon habitats. SalmonShield helps you visualize that connection and understand the environmental footprint of your street.

## What it does

- Takes your address and analyzes the area around it
- Shows nearby salmon streams on a map along with stormwater discharge points
- Calculates an environmental impact score based on things like road density, impervious surface coverage, water quality impairments, and storm drain density
- Gives you personalized suggestions for what you can actually do to help
- Displays precipitation data so you can see how much rain is hitting your area

## How to run it

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your API keys:

GOOGLE_MAPS_API_KEY=apikey
OPENAI_API_KEY=apikey


3. Start the server:

python app.py

4. Go to `http://localhost:5000` in your browser.

You can also test the results page directly with a lat/lon:
```
http://127.0.0.1:5000/results?lat=40.78&lon=-74.0
```

## Project structure

```
emphackfest-final/
├── app.py
├── requirements.txt
├── .env
├── data/
│   ├── salmon_streams.geojson
│   └── stormwater_discharge.geojson
├── static/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── index.js
│   │   └── results.js
│   └── media/
│       ├── logo.png
│       └── demo.mov
└── templates/
    ├── base.html
    ├── index.html
    └── results.html
```

## Stack

- **Flask** — backend server and routing
- **Leaflet.js** — interactive map
- **Google Maps Places API** — address autocomplete
- **OpenAI API** — generates personalized action suggestions
- **EPA / open government datasets** — water quality, stream, and stormwater data

## Built at

EmpHackfest 2026 by a team of students interested in environmental data and making geospatial information accessible to regular people.
