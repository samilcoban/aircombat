# ================================================
# FILE: src/core.py
# ================================================
from config import Config

# Facade Pattern: Select implementation based on configuration
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    # UPDATED: Removed dist_2d, bearing_deg
    from src.core_flat import AirCombatCore, Entity 
else:
    from src.core_geodetic import AirCombatCore, Entity