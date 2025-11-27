from config import Config

# Facade Pattern: Select implementation based on configuration
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    from src.env_flat import AirCombatEnv
else:
    # Default to Geodetic if not specified or set to 'curved'
    from src.env_geodetic import AirCombatEnv
