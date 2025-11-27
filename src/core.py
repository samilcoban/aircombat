from config import Config

# Facade Pattern: Select implementation based on configuration
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    from src.core_flat import AirCombatCore, Entity, dist_2d, bearing_deg
else:
    # Default to Geodetic if not specified or set to 'curved'
    from src.core_geodetic import AirCombatCore, Entity
    # Note: dist_2d and bearing_deg are NOT exported by core_geodetic
    # If code tries to import them from here while in geodetic mode, it will fail.
    # This is intended behavior as geodetic mode should use geodetic_distance_km etc.
