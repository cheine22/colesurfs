"""CSC — Colesurfs Correction.

A statistical post-processing layer that consumes GFS-Wave and ECMWF-WAM
forecasts at a buoy location plus seasonal features and emits a single
bias-corrected swell forecast. See the project plan for full design.

This package is intentionally decoupled from the Flask runtime —
training, backfill, and evaluation scripts run out-of-process. The Flask
app only imports `csc.serve` and `csc.predict` at inference time.
"""

__all__ = ["schema"]
