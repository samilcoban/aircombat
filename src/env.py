# ================================================
# FILE: src/env.py
# ================================================
"""
Gymnasium environment facade.

This module uses the Facade pattern to select the appropriate
environment implementation based on the PHYSICS_MODE configuration.

Supported modes:
- 'flat': Flat-earth environment (src/env_flat.py)
- 'curved'/'geodetic': Geodetic environment (src/env_geodetic.py)

The selected implementation provides AirCombatEnv, which is a
full Gymnasium environment with:
- Multi-agent observation/action spaces
- Graph-based state representation for GNN
- Configurable reward shaping
- Curriculum learning support
"""
from config import Config

# Facade Pattern: Select implementation based on configuration
# Intuition: This module acts as a switch to load either the flat-earth or geodetic environment.
# This allows the rest of the codebase to import 'AirCombatEnv' without worrying about the underlying model.
if hasattr(Config, 'PHYSICS_MODE') and Config.PHYSICS_MODE == 'flat':
    from src.env_flat import AirCombatEnv
else:
    # Default to Geodetic if not specified or set to 'curved'
    from src.env_geodetic import AirCombatEnv
