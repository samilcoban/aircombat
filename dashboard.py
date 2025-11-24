import streamlit as st
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

st.set_page_config(page_title="AirCombat 3.0 Dashboard", layout="wide")

st.title("✈️ AirCombat 3.0 Training Dashboard")

# --- Sidebar: Checkpoint Selection ---
st.sidebar.header("Controls")
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    st.error(f"Checkpoint directory '{checkpoint_dir}' not found!")
    st.stop()

# Find GIFs
gifs = sorted(glob.glob(os.path.join(checkpoint_dir, "*.gif")), key=os.path.getmtime, reverse=True)
if not gifs:
    st.warning("No validation GIFs found yet.")
else:
    selected_gif = st.sidebar.selectbox("Select Validation Replay", gifs)
    st.sidebar.text(f"Selected: {os.path.basename(selected_gif)}")

# --- Main Layout ---
# Top Row: Metrics
st.subheader("📈 Training Metrics")
col_m1, col_m2 = st.columns(2)

# Find TensorBoard Logs
log_dir = "runs"
latest_reward = 0.0
latest_loss = 0.0
df_reward = None
df_loss = None

if os.path.exists(log_dir):
    runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if runs:
        latest_run = max(runs, key=os.path.getmtime)
        
        try:
            ea = EventAccumulator(latest_run)
            ea.Reload()
            tags = ea.Tags()['scalars']
            
            if "charts/mean_step_reward" in tags:
                events = ea.Scalars("charts/mean_step_reward")
                steps = [e.step for e in events]
                values = [e.value for e in events]
                df_reward = pd.DataFrame({"Step": steps, "Reward": values})
                latest_reward = values[-1]
            
            if "charts/loss" in tags:
                events = ea.Scalars("charts/loss")
                steps = [e.step for e in events]
                values = [e.value for e in events]
                df_loss = pd.DataFrame({"Step": steps, "Loss": values})
                latest_loss = values[-1]
                
        except Exception as e:
            st.error(f"Error reading logs: {e}")

with col_m1:
    st.metric("Mean Reward", f"{latest_reward:.4f}")
    if df_reward is not None:
        st.line_chart(df_reward.set_index("Step"), height=300)

with col_m2:
    st.metric("Loss", f"{latest_loss:.4f}")
    if df_loss is not None:
        st.line_chart(df_loss.set_index("Step"), height=300)

st.markdown("---")

# Bottom Row: Replay & 3D Visualization
st.subheader("📺 Validation Replay & 3D View")

tab1, tab2 = st.tabs(["2D GIF Replay", "3D Interactive View"])

with tab1:
    if gifs:
        # Use a centered column for the video to keep it stable and not too huge
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(selected_gif, caption=os.path.basename(selected_gif), use_container_width=True)
    else:
        st.info("Waiting for first validation run...")

with tab2:
    # Find HTMLs
    htmls = sorted(glob.glob(os.path.join(checkpoint_dir, "*_3d.html")), key=os.path.getmtime, reverse=True)
    if htmls:
        selected_html = st.selectbox("Select 3D Visualization", htmls)
        with open(selected_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=False)
    else:
        st.info("No 3D visualizations found yet. They are generated every 50 updates.")

st.markdown("---")

# --- Action Statistics ---
st.subheader("🎮 Action Statistics")
if os.path.exists(log_dir):
    try:
        # Re-load EA if needed or reuse
        # (Assuming ea is available from above scope if try block succeeded)
        if 'ea' in locals():
            tags = ea.Tags()['scalars']
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                if "actions/fire_mean" in tags:
                    st.markdown("#### Missile Firing (Mean)")
                    events = ea.Scalars("actions/fire_mean")
                    df_fire = pd.DataFrame({"Step": [e.step for e in events], "Fire Action": [e.value for e in events]})
                    st.line_chart(df_fire.set_index("Step"), height=250)
            
            with col_a2:
                if "actions/g_pull_mean" in tags:
                    st.markdown("#### G-Pull (Mean)")
                    events = ea.Scalars("actions/g_pull_mean")
                    df_g = pd.DataFrame({"Step": [e.step for e in events], "G-Load": [e.value for e in events]})
                    st.line_chart(df_g.set_index("Step"), height=250)

    except Exception as e:
        st.warning(f"Could not load action stats: {e}")

st.caption("AirCombat 3.0 | RL Training Monitor")
