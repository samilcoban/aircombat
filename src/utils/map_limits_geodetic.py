# ================================================
# FILE: src/utils/map_limits_geodetic.py
# ================================================
"""
    Implements a latitude-longitude rectangle that defines the allowable region
    for a simulation.
    
    Intuition: In curved-earth (geodetic) mode, we use lat/lon coordinates. The Earth is not flat,
    so distances depend on the curvature of the ellipsoid (WGS84). This class handles the complexity
    of geodetic calculations while providing the same interface as the flat version.
"""

from geographiclib.geodesic import Geodesic
import numpy as np


class MapLimits:
    """
    Defines a rectangular lat/lon bounding box on Earth's surface.
    
    Math: The boundary is defined by [left_lon, right_lon] x [bottom_lat, top_lat].
    However, unlike flat space, the actual distances vary with latitude because meridians converge
    at the poles. We use WGS84 ellipsoid for accurate geodetic calculations.
    """
    
    def __init__(self, left_lon, bottom_lat, right_lon, top_lat):
        """
        Initialize the geodetic map boundaries.
        
        Args:
            left_lon: Western boundary in degrees
            bottom_lat: Southern boundary in degrees
            right_lon: Eastern boundary in degrees
            top_lat: Northern boundary in degrees
        """
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat

    def latitude_extent(self):
        """
        Calculate the span of latitude in degrees.
        
        Math: lat_extent = top_lat - bottom_lat
        Note: Each degree of latitude is approximately 111 km, regardless of longitude.
        """
        return self.top_lat - self.bottom_lat

    def longitude_extent(self):
        """
        Calculate the span of longitude in degrees.
        
        Math: lon_extent = right_lon - left_lon
        Note: The distance per degree of longitude varies with latitude:
              distance ≈ 111 km * cos(latitude)
              At the equator, 1° lon ≈ 111 km. At 60° lat, 1° lon ≈ 55.5 km.
        """
        return self.right_lon - self.left_lon

    def max_latitude_extent_km(self):
        """
        Calculate the maximum north-south extent in kilometers.
        
        Intuition: Computes the actual geodesic distance along the western and eastern
        boundaries and returns the maximum. This accounts for Earth's curvature.
        
        Math:
            d_west = geodesic_distance(bottom_lat, left_lon, top_lat, left_lon)
            d_east = geodesic_distance(bottom_lat, right_lon, top_lat, right_lon)
            return max(d_west, d_east)
        
        Uses WGS84 ellipsoid for accuracy.
        """
        # Distance along western edge
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.top_lat, self.left_lon,
                                    outmask=Geodesic.DISTANCE)
        # Distance along eastern edge
        d2 = Geodesic.WGS84.Inverse(self.bottom_lat, self.right_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        # Convert meters to kilometers and return the maximum
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def max_longitude_extent_km(self):
        """
        Calculate the maximum east-west extent in kilometers.
        
        Intuition: Computes the actual geodesic distance along the southern and northern
        boundaries and returns the maximum. The northern boundary is typically shorter
        because meridians converge toward the pole.
        
        Math:
            d_south = geodesic_distance(bottom_lat, left_lon, bottom_lat, right_lon)
            d_north = geodesic_distance(top_lat, left_lon, top_lat, right_lon)
            return max(d_south, d_north)
        
        Uses WGS84 ellipsoid for accuracy.
        """
        # Distance along southern edge
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.bottom_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        # Distance along northern edge
        d2 = Geodesic.WGS84.Inverse(self.top_lat, self.left_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        # Convert meters to kilometers and return the maximum
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def relative_position(self, lat, lon):
        """
        Convert absolute lat/lon to normalized relative [0,1] coordinates.
        
        Intuition: Neural networks prefer inputs in [0,1] range. This performs a simple
        linear normalization of lat/lon to [0,1], where (bottom_lat, left_lon) maps to (0,0)
        and (top_lat, right_lon) maps to (1,1).
        
        Math:
            lat_rel = (lat - bottom_lat) / (top_lat - bottom_lat)
            lon_rel = (lon - left_lon) / (right_lon - left_lon)
            Then clip to [0,1] to handle positions outside the boundary.
        
        NOTE: This is a linear approximation. It does NOT account for the varying distance
        per degree of longitude at different latitudes. For accurate distance-based normalization,
        you would need geodesic calculations. However, for bounded regions and RL observations,
        this linear approximation is usually sufficient.
        
        Returns:
            Tuple of (lat_rel, lon_rel) where both are in [0,1]
        """
        lat_rel = (lat - self.bottom_lat) / self.latitude_extent()
        lon_rel = (lon - self.left_lon) / self.longitude_extent()
        # Clip to [0,1] to gracefully handle positions slightly outside the boundary
        return np.clip(lat_rel, 0, 1), np.clip(lon_rel, 0, 1)

    def absolute_position(self, lat_rel, lon_rel):
        """
        Convert normalized relative [0,1] coordinates back to absolute lat/lon.
        
        Intuition: Reverse of relative_position(). Used when converting network outputs
        or normalized positions back to geodetic coordinates.
        
        Math:
            lat = lat_rel * (top_lat - bottom_lat) + bottom_lat
            lon = lon_rel * (right_lon - left_lon) + left_lon
        
        Returns:
            Tuple of (lat, lon) in degrees
        """
        lat = lat_rel * self.latitude_extent() + self.bottom_lat
        lon = lon_rel * self.longitude_extent() + self.left_lon
        return lat, lon

    def in_boundary(self, lat, lon):
        """
        Check if a point is within the map boundaries.
        
        Intuition: Used for boundary violation checks. Aircraft leaving the boundary
        typically receive penalties or the episode terminates.
        
        Math: Simple lat/lon bounding box containment test:
            left_lon <= lon <= right_lon AND bottom_lat <= lat <= top_lat
        
        Returns:
            True if point is inside or on the boundary, False otherwise
        """
        return self.left_lon <= lon <= self.right_lon and self.bottom_lat <= lat <= self.top_lat
