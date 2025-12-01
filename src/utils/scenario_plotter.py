import io
import math
from collections import namedtuple
from typing import List, Tuple, Optional
import matplotlib

matplotlib.use('Agg')
import cairo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

# A named tuple for RGBA colors for better readability.
ColorRGBA = namedtuple('ColorRGBA', ['red', 'green', 'blue', 'alpha'])


class Drawable:
    def __init__(self, zorder):
        self.zorder = zorder


class StatusMessage(Drawable):
    def __init__(self, text, text_color=ColorRGBA(1, 1, 1, 1), zorder: int = 100):
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class PolyLine(Drawable):
    def __init__(self, points: List[Tuple[float, float]], line_width: float = 1.0,
                 edge_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        super().__init__(zorder)
        self.points = points
        self.line_width = line_width
        self.edge_color = edge_color


class Airplane(Drawable):
    def __init__(self, x, y, heading, edge_color, fill_color, info_text=None, zorder=10):
        super().__init__(zorder)
        self.x = x
        self.y = y
        self.heading = heading
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.info_text = info_text


class Missile(Drawable):
    def __init__(self, x, y, heading, edge_color, fill_color, zorder=5):
        super().__init__(zorder)
        self.x = x
        self.y = y
        self.heading = heading
        self.edge_color = edge_color
        self.fill_color = fill_color


class ScenarioPlotter:
    def __init__(self, map_limits, dpi=100, width=800, height=800):
        self.base_limits = map_limits  # The hard map limits
        self.dpi = dpi
        self.width = width
        self.height = height

        # Dynamic Viewport State
        self.view_min_x = map_limits.min_x
        self.view_max_x = map_limits.max_x
        self.view_min_y = map_limits.min_y
        self.view_max_y = map_limits.max_y

    def _update_dynamic_viewport(self, objects):
        """
        Calculates a bounding box around all interesting entities (planes, missiles)
        and centers the view on them with padding.
        """
        xs = []
        ys = []

        has_entities = False
        for o in objects:
            if isinstance(o, (Airplane, Missile)):
                xs.append(o.x)
                ys.append(o.y)
                has_entities = True

        if not has_entities:
            return  # Keep previous view

        # 1. Find bounding box of combat
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 2. Add padding (buffer zone) - at least 2km or 20%
        span_x = max_x - min_x
        span_y = max_y - min_y

        # Minimum view size (e.g. 5km x 5km) to prevent extreme zoom on overlapping planes
        MIN_ZOOM = 5000.0

        target_span = max(max(span_x, span_y) * 1.5, MIN_ZOOM)

        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        # 3. Update view limits (preserving aspect ratio roughly)
        half_span = target_span / 2
        self.view_min_x = mid_x - half_span
        self.view_max_x = mid_x + half_span
        self.view_min_y = mid_y - half_span
        self.view_max_y = mid_y + half_span

    def _get_bg_surface(self):
        """Generates the grid background based on current dynamic view."""
        plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Convert meters to lat/lon for Cartopy (fake projection)
        # Assuming 0,0 is center. 111,000m = 1 degree
        m_deg = 111000.0

        # Set extent based on DYNAMIC view, not static map limits
        extent = [
            self.view_min_x / m_deg, self.view_max_x / m_deg,
            self.view_min_y / m_deg, self.view_max_y / m_deg
        ]

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.patch.set_facecolor('#191b24')

        # Dynamic Grid
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return cairo.ImageSurface.create_from_png(buf)

    def to_png(self, filename: str, objects: List[Drawable]):
        # 1. Calculate Auto-Zoom
        self._update_dynamic_viewport(objects)

        # 2. Create Background
        bg_surface = self._get_bg_surface()

        # 3. Setup Cairo
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.width, self.height)
        ctx = cairo.Context(surface)

        # Draw background scaled to fit
        ctx.save()
        scale_x = self.width / bg_surface.get_width()
        scale_y = self.height / bg_surface.get_height()
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(bg_surface, 0, 0)
        ctx.paint()
        ctx.restore()

        # 4. Coordinate Transform Helper
        # Maps Meter coordinates -> Pixel coordinates
        def to_screen(x_m, y_m):
            rel_x = (x_m - self.view_min_x) / (self.view_max_x - self.view_min_x)
            rel_y = (y_m - self.view_min_y) / (self.view_max_y - self.view_min_y)
            # Flip Y because Cairo 0,0 is top-left, but Map 0,0 is usually bottom-left or center
            # Actually, standard map: +Y is North (Up). Screen: +Y is Down.
            px = rel_x * self.width
            py = (1.0 - rel_y) * self.height
            return px, py

        # 5. Draw Objects
        for o in sorted(objects, key=lambda d: d.zorder):
            if isinstance(o, Airplane):
                px, py = to_screen(o.x, o.y)
                # Check if off-screen
                if 0 <= px <= self.width and 0 <= py <= self.height:
                    self._draw_plane(ctx, px, py, o)

            elif isinstance(o, Missile):
                px, py = to_screen(o.x, o.y)
                if 0 <= px <= self.width and 0 <= py <= self.height:
                    self._draw_missile(ctx, px, py, o)

            elif isinstance(o, StatusMessage):
                self._draw_text(ctx, 10, 20, o.text, o.text_color, size=14)

        surface.write_to_png(filename)

    def _draw_plane(self, ctx, x, y, o):
        ctx.save()
        ctx.translate(x, y)
        # Rotate: Math heading (0=North/Up, 90=East/Right).
        # Cairo: 0=Right, 90=Down.
        # Conversion: -Heading + 90 degrees
        angle = math.radians(-o.heading + 90)
        ctx.rotate(angle)

        # Triangle
        ctx.move_to(10, 0)
        ctx.line_to(-8, 6)
        ctx.line_to(-8, -6)
        ctx.close_path()

        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.set_line_width(1.5)
        ctx.stroke()

        # ID Text (Rotated back to be upright)
        ctx.rotate(-angle)
        self._draw_text(ctx, 15, -15, o.info_text, o.edge_color, size=11)
        ctx.restore()

    def _draw_missile(self, ctx, x, y, o):
        ctx.save()
        ctx.translate(x, y)
        angle = math.radians(-o.heading + 90)
        ctx.rotate(angle)

        ctx.set_source_rgba(*o.fill_color)
        ctx.rectangle(-4, -1, 8, 2)
        ctx.fill()
        ctx.restore()

    def _draw_text(self, ctx, x, y, text, color, size=12):
        if not text: return
        ctx.set_source_rgba(*color)
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(size)
        ctx.move_to(x, y)
        ctx.show_text(text)