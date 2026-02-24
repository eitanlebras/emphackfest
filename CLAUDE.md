# SalmonShield — Claude Code Instructions

## What This Project Is
SalmonShield is a civic web app built to pitch to Washington State government. It shows urban stormwater runoff impact on salmon streams. Users enter an address and see a danger score, environmental conditions, interactive map, and AI suggestions. This needs to look like a real government-ready product, not a student hackathon project.

## Tech Stack
- Backend: Flask (Python), GeoPandas, Shapely, OpenAI GPT-4o, python-dotenv
- Frontend: HTML, CSS, Leaflet.js, OpenStreetMap tiles
- Data: GeoJSON files served locally from /data folder
- Environment variables in .env — never touch or create .env

## Design System
- Background: #F7F9FC
- Primary: #1A6B9A
- Salmon accent: #E8603C (the coral/salmon red used for headings)
- High risk: #D94F3D
- Moderate risk: #E8A838
- Low risk: #4CAF7D
- Text: #1C2B3A
- Font: whatever is currently loaded — keep it, don't change fonts
- No gradients. Flat, clean, professional.
- Border radius 8px on cards. Subtle shadows: 0 2px 12px rgba(0,0,0,0.08)
- Never use generic bootstrap or tailwind utility soup — write clean custom CSS

## What Already Works
- Address search → Flask geocodes with Nominatim → returns danger score, risk level, environmental conditions
- Leaflet map with salmon streams (red/orange/green by risk), stormwater discharge points (purple), watershed boundaries, impervious surface heatmap layer
- Environmental conditions grid: water quality impairments, storm drains, heavy traffic EHD rank, road density, impervious surface %, precipitation forecast
- Stream click popups with species, risk, distance, watershed
- Layers toggle: watershed boundaries, impervious surface NLCD
- FAQ accordion
- "What You Can Do Here" section (AI suggestions pending API key)

## Rules
- Do not rewrite working code. Only add or modify what is asked.
- Do not change the color palette.
- Do not add new dependencies unless asked.
- Do not touch .env
- Write clean semantic HTML — no div soup
- CSS should be organized and commented
- Every change should make it look more like a real government product pitch
- When in doubt, do less and do it better