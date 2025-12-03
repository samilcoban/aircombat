# ================================================
# FILE: src/utils/angles.py
# ================================================
"""
    Angles computations
"""

import math

DEG_TO_RAD = math.pi / 180


def normalize_angle(a: float) -> float:
    """
    Normalizes an angle to the range [0, 360).
    
    Intuition: Headings like 370 degrees or -10 degrees are mathematically valid but
    inconvenient for lookups and comparisons. We wrap them to 0-360.
    """
    while a >= 360.0:
        a -= 360
    while a < 0.0:
        a += 360
    return a


def sum_angles(a: float, b: float) -> float:
    """
    Sums two angles and normalizes the result.
    """
    return normalize_angle(a + b)


def signed_heading_diff(actual: float, desired: float) -> float:
    """
    Calculates the shortest signed difference between two headings.
    Result is in range [-180, 180].
    
    Intuition: If I am at 10 deg and want to go to 350 deg, the difference is -20 deg (turn left),
    NOT +340 deg (turn right).
    
    Math:
    delta = desired - actual
    if delta < -180: delta += 360 (Wrap around left)
    if delta > 180: delta -= 360 (Wrap around right)
    """
    # actual and desired in [0, 360)
    delta = desired - actual
    if delta < -180:
        delta = 360 + delta
    if delta > 180:
        delta = -360 + delta
    return delta
