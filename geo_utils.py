"""
geo_utils.py
------------
Geographic utility functions reused throughout the project.
"""

from math import radians, sin, cos, sqrt, atan2, degrees
import numpy as np


def haversine(lat1: float, lon1: float,
              lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000.0  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2))
         * sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def bearing(lat1: float, lon1: float,
            lat2: float, lon2: float) -> float:
    """Bearing in degrees (0-360) between two GPS points."""
    dlon  = radians(lon2 - lon1)
    x     = sin(dlon) * cos(radians(lat2))
    y     = (cos(radians(lat1)) * sin(radians(lat2))
             - sin(radians(lat1)) * cos(radians(lat2)) * cos(dlon))
    return (degrees(atan2(x, y)) + 360) % 360


def normalize_position(lat: float, lon: float,
                        lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float
                        ) -> tuple[float, float]:
    """Normalize lat/lon to [0, 1] within the bounding box."""
    norm_lat = (lat - lat_min) / max(lat_max - lat_min, 1e-9)
    norm_lon = (lon - lon_min) / max(lon_max - lon_min, 1e-9)
    return (float(np.clip(norm_lat, 0, 1)),
            float(np.clip(norm_lon, 0, 1)))


def velocity_ms(lat1, lon1, lat2, lon2,
                dt_seconds: float) -> float:
    """Speed in m/s between two points."""
    if dt_seconds <= 0:
        return 0.0
    return haversine(lat1, lon1, lat2, lon2) / dt_seconds


def velocity_kmh(lat1, lon1, lat2, lon2,
                 dt_seconds: float) -> float:
    """Speed in km/h between two points."""
    return velocity_ms(lat1, lon1, lat2, lon2, dt_seconds) * 3.6
