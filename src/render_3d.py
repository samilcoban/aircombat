import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import Config
from aircombat_sim.utils.map_limits import MapLimits


class Render3D:
    """Interactive 3D visualization for aerial combat using Plotly."""
    
    def __init__(self, map_limits=None):
        self.cfg = Config
        if map_limits is None:
            self.map_limits = MapLimits(*self.cfg.MAP_LIMITS)
        else:
            self.map_limits = map_limits
        
        # Trajectory history
        self.trajectories = {}  # {uid: [(lat, lon, alt, time), ...]}
        self.max_trail_length = 50
        
    def reset(self):
        """Clear trajectory history."""
        self.trajectories = {}
    
    def update_trajectories(self, entities, time):
        """Update trajectory history for all entities."""
        for uid, ent in entities.items():
            if uid not in self.trajectories:
                self.trajectories[uid] = []
            
            self.trajectories[uid].append((ent.lat, ent.lon, ent.alt, time))
            
            # Keep only last N positions
            if len(self.trajectories[uid]) > self.max_trail_length:
                self.trajectories[uid].pop(0)
    
    def create_figure(self, entities, title="Aerial Combat 3D View"):
        """Create interactive 3D Plotly figure."""
        fig = go.Figure()
        
        # Separate entities by type and team
        blue_planes = []
        red_planes = []
        missiles = []
        
        for uid, ent in entities.items():
            if ent.type == "plane":
                if ent.team == "blue":
                    blue_planes.append((uid, ent))
                else:
                    red_planes.append((uid, ent))
            elif ent.type == "missile":
                missiles.append((uid, ent))
        
        # Add trajectories (trails)
        for uid, trail in self.trajectories.items():
            if len(trail) < 2:
                continue
            
            lats, lons, alts, times = zip(*trail)
            
            # Determine color based on entity
            if uid in entities:
                ent = entities[uid]
                color = 'rgba(0, 0, 255, 0.3)' if ent.team == "blue" else 'rgba(255, 0, 0, 0.3)'
                name = f"{ent.team.capitalize()} Trail"
            else:
                # Entity no longer exists (dead or missile expired)
                color = 'rgba(128, 128, 128, 0.2)'
                name = "Trail (Destroyed)"
            
            fig.add_trace(go.Scatter3d(
                x=lons,
                y=lats,
                z=alts,
                mode='lines',
                line=dict(color=color, width=2),
                name=name,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add Blue Planes
        for uid, ent in blue_planes:
            hover_text = (
                f"<b>Blue {uid}</b><br>"
                f"Alt: {int(ent.alt)}m<br>"
                f"Speed: {int(ent.speed)} kts<br>"
                f"Heading: {int(ent.heading)}°<br>"
                f"Fuel: {int(ent.fuel * 100)}%<br>"
                f"Ammo: {ent.ammo}<br>"
                f"G-Load: {ent.g_load:.1f}g"
            )
            
            fig.add_trace(go.Scatter3d(
                x=[ent.lon],
                y=[ent.lat],
                z=[ent.alt],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='blue',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                text=[f"B{uid}"],
                textposition="top center",
                textfont=dict(color='blue', size=10),
                name=f"Blue {uid}",
                hovertext=hover_text,
                hoverinfo='text'
            ))
            
            # Add velocity vector (heading indicator)
            self._add_velocity_vector(fig, ent, 'blue')
        
        # Add Red Planes
        for uid, ent in red_planes:
            hover_text = (
                f"<b>Red {uid}</b><br>"
                f"Alt: {int(ent.alt)}m<br>"
                f"Speed: {int(ent.speed)} kts<br>"
                f"Heading: {int(ent.heading)}°<br>"
                f"Fuel: {int(ent.fuel * 100)}%<br>"
                f"Ammo: {ent.ammo}<br>"
                f"G-Load: {ent.g_load:.1f}g"
            )
            
            fig.add_trace(go.Scatter3d(
                x=[ent.lon],
                y=[ent.lat],
                z=[ent.alt],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                text=[f"R{uid}"],
                textposition="top center",
                textfont=dict(color='red', size=10),
                name=f"Red {uid}",
                hovertext=hover_text,
                hoverinfo='text'
            ))
            
            # Add velocity vector
            self._add_velocity_vector(fig, ent, 'red')
        
        # Add Missiles
        for uid, ent in missiles:
            target_info = f"→ {ent.target_id}" if ent.target_id else "No Target"
            hover_text = (
                f"<b>Missile {uid}</b><br>"
                f"Target: {target_info}<br>"
                f"Alt: {int(ent.alt)}m<br>"
                f"Speed: {int(ent.speed)} kts<br>"
                f"Time: {ent.time_alive:.1f}s"
            )
            
            color = 'cyan' if ent.team == "blue" else 'orange'
            
            fig.add_trace(go.Scatter3d(
                x=[ent.lon],
                y=[ent.lat],
                z=[ent.alt],
                mode='markers',
                marker=dict(
                    size=6,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                name=f"Missile {uid}",
                hovertext=hover_text,
                hoverinfo='text'
            ))
            
            # Add missile velocity vector
            self._add_velocity_vector(fig, ent, color, scale=0.3)
        
        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Altitude (m)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                xaxis=dict(backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(backgroundcolor="rgb(200, 200,230)")
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            height=800,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    def _add_velocity_vector(self, fig, ent, color, scale=0.5):
        """Add velocity vector arrow to show heading and pitch."""
        # Calculate vector endpoint based on heading and pitch
        # Scale by speed for visual effect
        import math
        
        # Convert heading to radians
        heading_rad = math.radians(ent.heading)
        pitch_rad = ent.pitch if hasattr(ent, 'pitch') else 0.0
        
        # Vector length proportional to speed (scaled for visibility)
        vector_length = (ent.speed / 1000.0) * scale
        
        # Calculate 3D vector components
        dx = vector_length * math.sin(heading_rad) * math.cos(pitch_rad)
        dy = vector_length * math.cos(heading_rad) * math.cos(pitch_rad)
        dz = vector_length * math.sin(pitch_rad)
        
        # Add arrow
        fig.add_trace(go.Scatter3d(
            x=[ent.lon, ent.lon + dx],
            y=[ent.lat, ent.lat + dy],
            z=[ent.alt, ent.alt + dz],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def save_html(self, fig, filename):
        """Save figure as interactive HTML."""
        fig.write_html(filename)
        print(f"Saved 3D visualization to {filename}")
    
    def show(self, fig):
        """Display figure in browser."""
        fig.show()

    def create_animation(self, frames_data, filename):
        """
        Create animated 3D visualization from frame data.
        frames_data: List of dicts {'entities': {uid: EntityCopy}, 'time': t, 'step': s}
        """
        # Create frames for animation
        plotly_frames = []
        
        for i, frame_data in enumerate(frames_data):
            # Create figure for this frame
            # We temporarily override self.trajectories to match the frame time if needed
            # But for simplicity, we can just show the full trajectory or build it up.
            # Better: create_figure uses self.trajectories. 
            # If we want trails to grow, we'd need to manage trajectories per frame.
            # For now, let's just show the entities moving, with full trails or no trails?
            # Let's just show entities moving. Trails might be static or we can rebuild them.
            
            # To allow create_figure to work, we pass the entities from the frame
            fig = self.create_figure(
                frame_data['entities'],
                title=f"Step {frame_data.get('step', i)}, Time {frame_data['time']:.1f}s"
            )
            
            # Extract data for this frame
            plotly_frames.append(go.Frame(
                data=fig.data,
                name=str(i)
            ))
        
        if not frames_data:
            return

        # Create base figure from first frame
        base_fig = self.create_figure(
            frames_data[0]['entities'],
            title="Aerial Combat 3D Animation"
        )
        
        # Add frames to figure
        base_fig.frames = plotly_frames
        
        # Add play/pause buttons
        base_fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 100, "redraw": True},
                                         "fromcurrent": True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}])
                    ],
                    x=0.1,
                    y=0.0,
                    xanchor="left",
                    yanchor="bottom"
                )
            ],
            sliders=[{
                "active": 0,
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate"}],
                        "label": f"{frames_data[int(f.name)]['time']:.1f}s",
                        "method": "animate"
                    }
                    for f in plotly_frames
                ],
                "x": 0.1,
                "len": 0.9,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        base_fig.write_html(filename)
        print(f"Saved animated 3D visualization to {filename}")
