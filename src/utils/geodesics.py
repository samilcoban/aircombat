# ================================================
# FILE: src/utils/geodesics.py
# ================================================
"""
    Geodesics computations
"""

from typing import Tuple

from geographiclib.geodesic import Geodesic

# MODIFIED: Updated import path
from src.utils.angles import normalize_angle


def geodetic_distance_km(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """
    Calculates the geodesic distance (shortest path on ellipsoid) between two points.
    
    Math: Solves the inverse geodesic problem on the WGS84 ellipsoid.
    Returns: Distance in kilometers.
    """
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.DISTANCE)
    return r["s12"] / 1000.0


def geodetic_bearing_deg(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """
    Calculates the initial bearing (azimuth) from point 1 to point 2.
    
    Math: Solves the inverse geodesic problem for azimuth.
    Returns: Bearing in degrees [0, 360).
    """
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.AZIMUTH)
    return normalize_angle(r["azi1"])


def geodetic_direct(lat: float, lon: float, heading: float, distance: float) -> Tuple[float, float]:
    """
    Calculates the destination point given a start point, heading, and distance.
    
    Math: Solves the direct geodesic problem on the WGS84 ellipsoid.
    Args:
        distance: Distance in meters.
    Returns: (Latitude, Longitude) of destination.
    """
    d = Geodesic.WGS84.Direct(lat, lon, heading, distance, outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE)
    return d["lat2"], d["lon2"]