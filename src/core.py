# ================================================
# FILE: src/core.py
# ================================================
"""
Core physics simulation facade.

This module uses the Facade pattern to select the appropriate
physics implementation based on the PHYSICS_MODE configuration.

Supported modes:
- 'flat': Simplified flat-earth physics (faster, suitable for training)
- 'curved'/'geodetic': Full geodetic physics with Earth curvature

The selected implementation provides:
- AirCombatCore: The main physics simulation engine
- Entity: Individual aircraft/missile entity class
"""
from config import Config

# Facade Pattern: Select implementation based on configuration
# This allows the rest of the codebase to import from src.core
# without worrying about which physics model is active.
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    # Flat-earth mode: Uses Cartesian coordinates and simplified physics.
    # UPDATED: Removed dist_2d, bearing_deg
    from src.core_flat import AirCombatCore, Entity 
else:
    # Geodetic mode: Uses latitude/longitude and accounts for Earth curvature.
    from src.core_geodetic import AirCombatCore, Entity