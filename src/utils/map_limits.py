from config import Config

# Facade Pattern: Select implementation based on configuration
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    from src.utils.map_limits_flat import MapLimits
else:
    # Default to Geodetic if not specified or set to 'curved'
    from src.utils.map_limits_geodetic import MapLimits
