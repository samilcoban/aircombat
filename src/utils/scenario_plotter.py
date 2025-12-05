# ================================================
# FILE: src/utils/scenario_plotter.py
# ================================================
"""
    2D Scenario Visualization using Cairo and Matplotlib.

    UPDATED:
    1. Handles Radian -> Cairo Rotation conversion.
    2. Ensures visual consistency with the new Physics Core.
"""

import io
import math
from collections import namedtuple
from typing import List, Tuple, Optional
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import cairo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

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
        self.heading = heading  # Now in RADIANS
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.info_text = info_text


class Missile(Drawable):
    def __init__(self, x, y, heading, edge_color, fill_color, zorder=5):
        super().__init__(zorder)
        self.x = x
        self.y = y
        self.heading = heading  # Now in RADIANS
        self.edge_color = edge_color
        self.fill_color = fill_color


class ScenarioPlotter:
    def __init__(self, map_limits, dpi=100, width=800, height=800):
        self.base_limits = map_limits
        self.dpi = dpi
        self.width = width
        self.height = height

        # Dynamic Viewport State
        self.view_min_x = map_limits.min_x
        self.view_max_x = map_limits.max_x
        self.view_min_y = map_limits.min_y
        self.view_max_y = map_limits.max_y

    def _update_dynamic_viewport(self, objects):
        xs = []
        ys = []
        has_entities = False

        for o in objects:
            if isinstance(o, (Airplane, Missile)):
                xs.append(o.x)
                ys.append(o.y)
                has_entities = True

        if not has_entities: return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = max_x - min_x
        span_y = max_y - min_y

        # Minimum zoom 5km to keep context
        MIN_ZOOM = 5000.0
        target_span = max(max(span_x, span_y) * 1.5, MIN_ZOOM)

        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        half_span = target_span / 2
        self.view_min_x = mid_x - half_span
        self.view_max_x = mid_x + half_span
        self.view_min_y = mid_y - half_span
        self.view_max_y = mid_y + half_span

    def _get_bg_surface(self):
        plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Pseudo-projection for flat map
        m_deg = 111000.0
        extent = [
            self.view_min_x / m_deg, self.view_max_x / m_deg,
            self.view_min_y / m_deg, self.view_max_y / m_deg
        ]

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.patch.set_facecolor('#191b24')

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return cairo.ImageSurface.create_from_png(buf)

    def to_png(self, filename: str, objects: List[Drawable]):
        self._update_dynamic_viewport(objects)
        bg_surface = self._get_bg_surface()

        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.width, self.height)
        ctx = cairo.Context(surface)

        # Draw Background
        ctx.save()
        scale_x = self.width / bg_surface.get_width()
        scale_y = self.height / bg_surface.get_height()
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(bg_surface, 0, 0)
        ctx.paint()
        ctx.restore()

        # Coordinate Transform
        def to_screen(x_m, y_m):
            rel_x = (x_m - self.view_min_x) / (self.view_max_x - self.view_min_x)
            rel_y = (y_m - self.view_min_y) / (self.view_max_y - self.view_min_y)
            px = rel_x * self.width
            py = (1.0 - rel_y) * self.height  # Flip Y for screen coords
            return px, py

        # Draw Objects
        for o in sorted(objects, key=lambda d: d.zorder):
            if isinstance(o, Airplane):
                px, py = to_screen(o.x, o.y)
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

        # ROTATION CORRECTION (Radians -> Cairo)
        # Sim 0 = North (+Y). Sim PI/2 = East (+X).
        # Cairo 0 = Right (+X). Cairo -PI/2 = Up (-Y).
        # We assume Screen Y is flipped relative to Map Y.
        # Screen: Up is -Y.

        # Math:
        # If Sim Heading = 0 (North/Up). On screen, that's pointing to -Y (-90 deg).
        # If Sim Heading = PI/2 (East/Right). On screen, that's +X (0 deg).
        # Transformation: cairo_angle = heading - PI/2

        angle = o.heading - (math.pi / 2.0)
        ctx.rotate(angle)

        # Draw Triangle (Pointing Right/East is "Forward" in Cairo 0-deg frame)
        ctx.move_to(10, 0)  # Nose
        ctx.line_to(-8, 6)  # Bottom Tail
        ctx.line_to(-8, -6)  # Top Tail
        ctx.close_path()

        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.set_line_width(1.5)
        ctx.stroke()

        # Text (Undo rotation so it's readable)
        ctx.rotate(-angle)
        self._draw_text(ctx, 15, -15, o.info_text, o.edge_color, size=11)
        ctx.restore()

    def _draw_missile(self, ctx, x, y, o):
        ctx.save()
        ctx.translate(x, y)

        # Same rotation logic
        angle = o.heading - (math.pi / 2.0)
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