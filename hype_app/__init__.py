"""hype_app — support modules for the hyporheic Shiny web app.

geometry  — drawn GeoJSON -> projected GeoDataFrames
dem       — USGS 3DEP DEM fetch for the drawn domain
estimate  — pre-run grid-size estimate + guardrail bands
run       — assemble + execute a run_hyporheic call (in a worker thread)
results   — run artifacts -> map-ready GeoJSON + summary
bundle    — zip a run's outputs for download
"""
