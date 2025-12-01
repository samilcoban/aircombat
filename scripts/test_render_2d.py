import sys
import os
import torch
import imageio
import numpy as np

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import AirCombatEnv
from src.model import AgentTransformer
from src.utils.scenario_plotter import ScenarioPlotter, Airplane, Missile, StatusMessage, ColorRGBA
from config import Config


def test_render(checkpoint_path):
    print(f"Testing 2D Render with {checkpoint_path}...")

    # 1. Load Model
    model = AgentTransformer().to(Config.DEVICE)
    try:
        ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    except:
        print("Could not load checkpoint, using random weights.")
    model.eval()

    # 2. Setup Env & Plotter
    env = AirCombatEnv()
    # Use full map limits for the plotter base
    plotter = ScenarioPlotter(env.map_limits, dpi=100, width=600, height=600)

    obs, info = env.reset()
    frames = []

    print("Simulating episode...")
    for i in range(200):  # Run 200 steps
        # Get Action
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            action, _, _, _, _ = model.get_action_and_value(obs_t)

        # Step
        # Add a dummy opponent action just in case
        blue_act = action.cpu().numpy().flatten()
        if "red_obs" in info:
            # Just fly straight for red to test visuals
            red_act = np.array([0, 0, 0.5, 0, 0])
            full_act = np.concatenate([blue_act, red_act])
            obs, _, done, trunc, info = env.step(full_act)
        else:
            obs, _, done, trunc, info = env.step(blue_act)

        # 3. RENDER FRAME
        if i % 2 == 0:  # Every 2nd frame
            objects = []
            core = env.core

            # Blue Plane
            for uid in env.blue_ids:
                if uid in core.entities:
                    e = core.entities[uid]
                    objects.append(Airplane(e.x, e.y, e.heading,
                                            edge_color=ColorRGBA(0, 1, 1, 1),  # Cyan
                                            fill_color=ColorRGBA(0, 0.5, 0.5, 0.5),
                                            info_text=f"Blue (Alt: {int(e.alt)})"))

            # Red Plane
            for uid in env.red_ids:
                if uid in core.entities:
                    e = core.entities[uid]
                    objects.append(Airplane(e.x, e.y, e.heading,
                                            edge_color=ColorRGBA(1, 0, 0, 1),  # Red
                                            fill_color=ColorRGBA(0.5, 0, 0, 0.5),
                                            info_text=f"Red"))

            # Missiles
            for e in core.entities.values():
                if e.type == "missile":
                    objects.append(Missile(e.x, e.y, e.heading,
                                           edge_color=ColorRGBA(1, 1, 0, 1),
                                           fill_color=ColorRGBA(1, 1, 0, 1)))

            # HUD
            objects.append(StatusMessage(f"Step: {i} | Time: {core.time:.1f}s", zorder=100))

            fname = f"temp_render_{i:03d}.png"
            plotter.to_png(fname, objects)
            frames.append(fname)

        if done or trunc:
            break

    # 4. Save GIF
    if frames:
        out_path = "test_render.gif"
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(out_path, images, fps=15)
        print(f"✅ Render successful! Saved to {out_path}")

        # Cleanup
        for f in frames:
            if os.path.exists(f): os.remove(f)
    else:
        print("❌ No frames generated.")


if __name__ == "__main__":
    # Use latest if available, else standard name
    ckpt = "checkpoints/model_latest.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]

    test_render(ckpt)