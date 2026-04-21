"""CSC2 — Colesurfs Correction v2.

Fresh-start model architecture: corrects EURO (CMEMS) and GFS (Open-Meteo)
swell-partition forecasts against NDBC buoy observations, predicting primary
and secondary swell height / period / direction at eight buoys.

Eastern track (user-facing): 44013, 44065, 44097, 44091, 44098
Western track (silent):      46025, 46221, 46268

All forecast inputs MUST pass through the same processing pipeline as the
main dashboard (waves_cmems + waves), so downstream display stays identical.
The live-collection loop is in csc2.logger; schema is in csc2.schema.
"""
