# ================================================
# FILE: src/utils/map_limits_flat.py
# ================================================
"""
    Implements a Cartesian rectangle that defines the allowable region
    for a simulation (North-East-Down).
    
    Intuition: In flat-earth mode, we use a simple 2D Cartesian coordinate system.
    This is like a rectangular grid where x=East, y=North (NED convention).
    This class provides boundary checks and coordinate transformations.
"""

import numpy as np


class MapLimits:
    """
    Defines a rectangular boundary in flat Cartesian space.
    
    Math: The boundary is defined by [min_x, max_x] x [min_y, max_y].
    This creates a simple axis-aligned bounding box (AABB).
    """
    
    def __init__(self, min_x, max_x, min_y, max_y):
        """
        Initialize the rectangular map boundaries.
        
        Args:
            min_x: Minimum x coordinate (westernmost point in meters)
            max_x: Maximum x coordinate (easternmost point in meters)
            min_y: Minimum y coordinate (southernmost point in meters)
            max_y: Maximum y coordinate (northernmost point in meters)
        """
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def x_extent(self):
        """
        Calculate the width of the map in meters.
        
        Math: width = max_x - min_x
        """
        return self.max_x - self.min_x

    def y_extent(self):
        """
        Calculate the height of the map in meters.
        
        Math: height = max_y - min_y
        """
        return self.max_y - self.min_y

    def relative_position(self, x, y):
        """
        Convert absolute x,y coordinates to normalized relative [0,1] coordinates.
        
        Intuition: Neural networks prefer inputs in [0,1] or [-1,1] range. This normalizes
        position so that the bottom-left corner is (0,0) and top-right is (1,1).
        
        Math:
            x_rel = (x - min_x) / (max_x - min_x)
            y_rel = (y - min_y) / (max_y - min_y)
            Then clip to [0,1] to handle out-of-bounds positions gracefully.
        
        Returns:
            Tuple of (x_rel, y_rel) where both are in [0,1]
        """
        x_rel = (x - self.min_x) / self.x_extent()
        y_rel = (y - self.min_y) / self.y_extent()
        # Clip to [0,1] to handle positions slightly outside the boundary
        return np.clip(x_rel, 0, 1), np.clip(y_rel, 0, 1)

    def absolute_position(self, x_rel, y_rel):
        """
        Convert normalized relative [0,1] coordinates back to absolute x,y meters.
        
        Intuition: Reverse of relative_position(). Used when we need to convert
        network outputs or normalized positions back to real-world coordinates.
        
        Math:
            x = x_rel * (max_x - min_x) + min_x
            y = y_rel * (max_y - min_y) + min_y
        
        Returns:
            Tuple of (x, y) in meters
        """
        x = x_rel * self.x_extent() + self.min_x
        y = y_rel * self.y_extent() + self.min_y
        return x, y

    def in_boundary(self, x, y):
        """
        Check if a point is within the map boundaries.
        
        Intuition: Used for boundary violation checks. Aircraft leaving the boundary
        typically receive penalties or the episode terminates.
        
        Math: Simple AABB (Axis-Aligned Bounding Box) containment test:
            min_x <= x <= max_x AND min_y <= y <= max_y
        
        Returns:
            True if point is inside or on the boundary, False otherwise
        """
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
