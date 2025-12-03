# ================================================
# FILE: src/utils/scenario_plotter.py
# ================================================
"""
    2D Scenario Visualization using Cairo and Matplotlib.
    
    Intuition: This module renders air combat scenarios to PNG images. It combines:
    1. Matplotlib + Cartopy for grid backgrounds (geographic context)
    2. Cairo for fast, high-quality vector drawing of aircraft and missiles
    
    Key feature: DYNAMIC VIEWPORT - automatically zooms to follow the action,
    so you always see the combat clearly, not empty space.
"""

import io
import math
from collections import namedtuple
from typing import List, Tuple, Optional
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server/batch rendering
import cairo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

# A named tuple for RGBA colors for better readability.
# Each component is a float in [0,1] range.
ColorRGBA = namedtuple('ColorRGBA', ['red', 'green', 'blue', 'alpha'])


class Drawable:
    """
    Base class for all drawable objects.
    
    Zorder determines drawing order (lower = drawn first, higher = drawn on top).
    This is like layers in Photoshop.
    """
    def __init__(self, zorder):
        self.zorder = zorder


class StatusMessage(Drawable):
    """
    Text overlay for displaying information (e.g., "Blue wins!", "Iteration 1234").
    
    Drawn at high zorder (100) to ensure it appears on top of all other elements.
    """
    def __init__(self, text, text_color=ColorRGBA(1, 1, 1, 1), zorder: int = 100):
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class PolyLine(Drawable):
    """
    A series of connected line segments (trajectory trail).
    
    Intuition: Used to draw flight paths showing where aircraft have been.
    Low zorder (0) means drawn behind aircraft/missiles.
    """
    def __init__(self, points: List[Tuple[float, float]], line_width: float = 1.0,
                 edge_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        super().__init__(zorder)
        self.points = points  # List of (x, y) coordinates
        self.line_width = line_width
        self.edge_color = edge_color


class Airplane(Drawable):
    """
    Represents an aircraft to be rendered.
    
    Attributes:
        x, y: Position in meters (or lat/lon depending on mode)
        heading: Direction in degrees (0=North, 90=East)
        edge_color: Outline color
        fill_color: Interior color
        info_text: Optional label (e.g., "Blue 1", "Red 2")
    """
    def __init__(self, x, y, heading, edge_color, fill_color, info_text=None, zorder=10):
        super().__init__(zorder)
        self.x = x
        self.y = y
        self.heading = heading  # Degrees, 0=North
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.info_text = info_text


class Missile(Drawable):
    """
    Represents a missile to be rendered.
    
    Similar to Airplane but simpler geometry (just a rectangle).
    Drawn at zorder=5, between trails (0) and aircraft (10).
    """
    def __init__(self, x, y, heading, edge_color, fill_color, zorder=5):
        super().__init__(zorder)
        self.x = x
        self.y = y
        self.heading = heading  # Degrees, 0=North
        self.edge_color = edge_color
        self.fill_color = fill_color


class ScenarioPlotter:
    """
    Main rendering engine for air combat scenarios.
    
    Workflow:
    1. Calculate dynamic viewport (auto-zoom to action)
    2. Generate grid background using Matplotlib + Cartopy
    3. Draw entities (trails, missiles, aircraft, text) using Cairo
    4. Export to PNG
    
    Math: Performs coordinate transformations from world space (meters or lat/lon)
    to screen space (pixels).
    """
    
    def __init__(self, map_limits, dpi=100, width=800, height=800):
        """
        Initialize the plotter.
        
        Args:
            map_limits: MapLimits instance (defines the hard boundaries)
            dpi: Dots per inch (affects text/line quality)
            width, height: Output image dimensions in pixels
        """
        self.base_limits = map_limits  # The hard map limits (max possible area)
        self.dpi = dpi
        self.width = width
        self.height = height

        # Dynamic Viewport State (initially set to full map, then auto-adjusts)
        self.view_min_x = map_limits.min_x
        self.view_max_x = map_limits.max_x
        self.view_min_y = map_limits.min_y
        self.view_max_y = map_limits.max_y

    def _update_dynamic_viewport(self, objects):
        """
        Calculates a bounding box around all interesting entities (planes, missiles)
        and centers the view on them with padding.
        
        Intuition: Instead of showing the entire map (which might be 50km x 50km),
        zoom to where the action is happening. This is like a sports broadcast camera
        that follows the ball.
        
        Math:
        1. Find bounding box: [min_x, max_x] x [min_y, max_y] of all entities
        2. Expand by 50% (factor of 1.5) to add breathing room
        3. Enforce minimum zoom (5km) to prevent extreme closeup on overlapping planes
        4. Center viewport on the midpoint
        """
        xs = []
        ys = []

        has_entities = False
        for o in objects:
            # Only consider aircraft and missiles for viewport calculation
            # (ignore trails and text)
            if isinstance(o, (Airplane, Missile)):
                xs.append(o.x)
                ys.append(o.y)
                has_entities = True

        if not has_entities:
            return  # Keep previous view (no entities to track)

        # 1. Find bounding box of combat
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 2. Add padding (buffer zone) - at least 2km or 20%
        span_x = max_x - min_x
        span_y = max_y - min_y

        # Minimum view size (e.g. 5km x 5km) to prevent extreme zoom on overlapping planes
        MIN_ZOOM = 5000.0  # meters

        # Take the larger of: (current span * 1.5) or MIN_ZOOM
        # This ensures we always have some context around the entities
        target_span = max(max(span_x, span_y) * 1.5, MIN_ZOOM)

        # 3. Calculate center point
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        # 4. Update view limits (preserving square aspect ratio)
        half_span = target_span / 2
        self.view_min_x = mid_x - half_span
        self.view_max_x = mid_x + half_span
        self.view_min_y = mid_y - half_span
        self.view_max_y = mid_y + half_span

    def _get_bg_surface(self):
        """
        Generates the grid background based on current dynamic view.
        
        Intuition: Uses Matplotlib + Cartopy to create a professional-looking map background
        with grid lines. The background adapts to the current viewport.
        
        Math: Converts meters to pseudo lat/lon for Cartopy rendering:
            1 degree ≈ 111 km = 111,000 meters
        This is a rough approximation, but fine for visualization.
        
        Returns:
            Cairo ImageSurface containing the background
        """
        # Create matplotlib figure
        plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())  # Flat lat/lon projection

        # Convert meters to lat/lon for Cartopy (fake projection for flat mode)
        # Assuming 0,0 is center. 111,000m = 1 degree latitude
        m_deg = 111000.0

        # Set extent based on DYNAMIC view, not static map limits
        # extent = [lon_min, lon_max, lat_min, lat_max]
        extent = [
            self.view_min_x / m_deg, self.view_max_x / m_deg,
            self.view_min_y / m_deg, self.view_max_y / m_deg
        ]

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.patch.set_facecolor('#191b24')  # Dark blue-gray background

        # Add grid lines (auto-spaced based on view extent)
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Render to PNG in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()  # Free memory
        buf.seek(0)  # Reset buffer to beginning
        return cairo.ImageSurface.create_from_png(buf)

    def to_png(self, filename: str, objects: List[Drawable]):
        """
        Main rendering method. Generates a PNG image from a list of drawable objects.
        
        Workflow:
        1. Auto-zoom to action (update viewport)
        2. Generate grid background
        3. Draw all objects in zorder
        4. Save to file
        
        Args:
            filename: Output path (e.g., "frame_0001.png")
            objects: List of Drawable instances (Airplane, Missile, PolyLine, StatusMessage)
        """
        # 1. Calculate Auto-Zoom based on entity positions
        self._update_dynamic_viewport(objects)

        # 2. Create Background (matplotlib grid)
        bg_surface = self._get_bg_surface()

        # 3. Setup Cairo rendering context
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.width, self.height)
        ctx = cairo.Context(surface)

        # Draw background scaled to fit canvas
        ctx.save()
        scale_x = self.width / bg_surface.get_width()
        scale_y = self.height / bg_surface.get_height()
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(bg_surface, 0, 0)
        ctx.paint()  # Composite background onto canvas
        ctx.restore()

        # 4. Coordinate Transform Helper
        # Maps World coordinates (meters) -> Screen coordinates (pixels)
        def to_screen(x_m, y_m):
            """
            Transform world-space coordinates to pixel coordinates.
            
            Math:
            1. Normalize to [0,1] relative to viewport:
                rel_x = (x - view_min_x) / (view_max_x - view_min_x)
                rel_y = (y - view_min_y) / (view_max_y - view_min_y)
            
            2. Map to pixel space:
                px = rel_x * width
                py = (1 - rel_y) * height  # Flip Y axis
            
            Y-flip explanation:
            - Map convention: +Y = North (upward)
            - Screen convention: +Y = Down (from top-left origin)
            - So we flip: py = (1 - rel_y) * height
            """
            rel_x = (x_m - self.view_min_x) / (self.view_max_x - self.view_min_x)
            rel_y = (y_m - self.view_min_y) / (self.view_max_y - self.view_min_y)
            px = rel_x * self.width
            py = (1.0 - rel_y) * self.height  # Flip Y: map up = screen up
            return px, py

        # 5. Draw Objects in zorder (painters algorithm: back to front)
        # Sort by zorder: trails (0) -> missiles (5) -> planes (10) -> text (100)
        for o in sorted(objects, key=lambda d: d.zorder):
            if isinstance(o, Airplane):
                px, py = to_screen(o.x, o.y)
                # Check if off-screen (culling optimization)
                if 0 <= px <= self.width and 0 <= py <= self.height:
                    self._draw_plane(ctx, px, py, o)

            elif isinstance(o, Missile):
                px, py = to_screen(o.x, o.y)
                if 0 <= px <= self.width and 0 <= py <= self.height:
                    self._draw_missile(ctx, px, py, o)

            elif isinstance(o, StatusMessage):
                # Text is drawn at fixed screen position (top-left)
                self._draw_text(ctx, 10, 20, o.text, o.text_color, size=14)

        # 6. Save to file
        surface.write_to_png(filename)

    def _draw_plane(self, ctx, x, y, o):
        """
        Draw an aircraft as a triangle pointing in its heading direction.
        
        Math: Coordinate system conversion:
        - Aviation: 0° = North (up), 90° = East (right), clockwise
        - Cairo: 0° = East (right), 90° = South (down), clockwise
        - Conversion: cairo_angle = -aviation_heading + 90°
        
        Geometry: Isosceles triangle with nose pointing forward.
        """
        ctx.save()  # Save current transformation state
        ctx.translate(x, y)  # Move origin to aircraft position
        
        # Rotate to heading
        # Cairo rotation: 0° = Right, 90° = Down (clockwise)
        # Aviation heading: 0° = North/Up, 90° = East/Right (clockwise)
        # Conversion formula: -heading + 90°
        angle = math.radians(-o.heading + 90)
        ctx.rotate(angle)

        # Draw triangle pointing right (before rotation, this is the "forward" direction)
        # Nose at (10, 0), tail at (-8, ±6)
        ctx.move_to(10, 0)   # Nose
        ctx.line_to(-8, 6)   # Bottom tail
        ctx.line_to(-8, -6)  # Top tail
        ctx.close_path()     # Connect back to nose

        # Fill interior
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()  # Fill but keep path for stroke
        
        # Draw outline
        ctx.set_source_rgba(*o.edge_color)
        ctx.set_line_width(1.5)
        ctx.stroke()

        # Draw ID Text (rotate back to be upright/readable)
        ctx.rotate(-angle)  # Undo rotation so text is horizontal
        self._draw_text(ctx, 15, -15, o.info_text, o.edge_color, size=11)
        ctx.restore()  # Restore transformation state

    def _draw_missile(self, ctx, x, y, o):
        """
        Draw a missile as a small rotated rectangle.
        
        Simpler than aircraft - just a filled rectangle, no outline or label.
        Uses same heading conversion as aircraft.
        """
        ctx.save()
        ctx.translate(x, y)
        # Same heading conversion as aircraft
        angle = math.radians(-o.heading + 90)
        ctx.rotate(angle)

        # Draw a small rectangle pointing right (8px long, 2px wide)
        ctx.set_source_rgba(*o.fill_color)
        ctx.rectangle(-4, -1, 8, 2)  # Centered on origin
        ctx.fill()
        ctx.restore()

    def _draw_text(self, ctx, x, y, text, color, size=12):
        """
        Draw text at the specified screen coordinates.
        
        Args:
            ctx: Cairo context
            x, y: Screen pixel coordinates (not world coordinates)
            text: String to render
            color: ColorRGBA tuple
            size: Font size in points
        """
        if not text: return  # Skip if empty
        ctx.set_source_rgba(*color)  # Unpack RGBA tuple
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(size)
        ctx.move_to(x, y)  # Position cursor
        ctx.show_text(text)  # Render text