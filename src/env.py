# ================================================
# FILE: src/env.py
# ================================================
from config import Config

# Facade Pattern: Select implementation based on configuration
# Intuition: This module acts as a switch to load either the flat-earth or geodetic environment.
# This allows the rest of the codebase to import 'AirCombatEnv' without worrying about the underlying model.
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    from src.env_flat import AirCombatEnv
else:
    # Default to Geodetic if not specified or set to 'curved'
    from src.env_geodetic import AirCombatEnv
