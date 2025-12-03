# ================================================
# FILE: src/utils/map_limits.py
# ================================================
from config import Config

# Facade Pattern: Select implementation based on configuration
# Intuition: The rest of the codebase shouldn't care whether we are on a flat earth or a round earth.
# They just import 'MapLimits' from here, and this file handles the switch.
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    from src.utils.map_limits_flat import MapLimits
else:
    # Default to Geodetic if not specified or set to 'curved'
    from src.utils.map_limits_geodetic import MapLimits
