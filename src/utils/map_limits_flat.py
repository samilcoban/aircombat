"""
    Implements a Cartesian rectangle that defines the allowable region
    for a simulation (North-East-Down).
"""

import numpy as np


class MapLimits:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def x_extent(self):
        return self.max_x - self.min_x

    def y_extent(self):
        return self.max_y - self.min_y

    def relative_position(self, x, y):
        """Convert absolute x,y to relative [0,1] coordinates."""
        x_rel = (x - self.min_x) / self.x_extent()
        y_rel = (y - self.min_y) / self.y_extent()
        return np.clip(x_rel, 0, 1), np.clip(y_rel, 0, 1)

    def absolute_position(self, x_rel, y_rel):
        """Convert relative [0,1] coordinates to absolute x,y."""
        x = x_rel * self.x_extent() + self.min_x
        y = y_rel * self.y_extent() + self.min_y
        return x, y

    def in_boundary(self, x, y):
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
