import streamlit as st
import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

st.set_page_config(page_title="AirCombat 3.0 Dashboard", layout="wide")
st.title("✈️ AirCombat 3.0 Training Dashboard")

# --- Sidebar: Checkpoint Selection ---
st.sidebar.header("Controls")
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    st.error(f"Checkpoint directory '{checkpoint_dir}' not found!")
    st.stop()

gifs = sorted(glob.glob(os.path.join(checkpoint_dir, "*.gif")), key=os.path.getmtime, reverse=True)
if not gifs:
    st.sidebar.warning("No validation GIFs found yet.")
else:
    selected_gif = st.sidebar.selectbox("Select Validation Replay", gifs)
    st.sidebar.text(f"Selected: {os.path.basename(selected_gif)}")

# --- Main Layout ---
st.subheader("📈 Training Metrics")

log_dir = "runs"
latest_reward = 0.0
latest_loss = 0.0
df_reward = None
df_loss = None
df_components = pd.DataFrame()

if os.path.exists(log_dir):
    runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if runs:
        latest_run = max(runs, key=os.path.getmtime)
        try:
            ea = EventAccumulator(latest_run)
            ea.Reload()
            tags = ea.Tags()['scalars']


            # Helper to get scalar df
            def get_df(tag):
                if tag not in tags: return None
                events = ea.Scalars(tag)
                return pd.DataFrame({"Step": [e.step for e in events], "Value": [e.value for e in events]})


            # Main Metrics
            df_reward = get_df("train/reward")
            if df_reward is not None: latest_reward = df_reward.iloc[-1]["Value"]

            df_loss = get_df("train/loss")
            if df_loss is not None: latest_loss = df_loss.iloc[-1]["Value"]

            # Reward Components
            comp_tags = {
                "Kill": "rewards/rew_kill",
                "Pos": "rewards/rew_pos",
                "Penalty": "rewards/rew_penalty",
                "Survival": "rewards/rew_survival"
            }

            comp_data = {}
            for name, tag in comp_tags.items():
                d = get_df(tag)
                if d is not None:
                    if "Step" not in comp_data: comp_data["Step"] = d["Step"]
                    comp_data[name] = d["Value"]

            if comp_data:
                df_components = pd.DataFrame(comp_data).set_index("Step")

        except Exception as e:
            st.error(f"Error reading logs: {e}")

# Row 1: High Level
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Mean Reward", f"{latest_reward:.4f}")
    if df_reward is not None: st.line_chart(df_reward.set_index("Step"), height=250)

with col_m2:
    st.metric("Total Loss", f"{latest_loss:.4f}")
    if df_loss is not None: st.line_chart(df_loss.set_index("Step"), height=250)

# Row 2: Components
st.subheader("📊 Reward Composition")
if not df_components.empty:
    st.line_chart(df_components, height=300)
    st.caption("Breakdown: Kill (Terminal), Pos (Shaping), Penalty (Crashing/Time), Survival (Energy)")
else:
    st.info("No reward component logs found yet.")

st.markdown("---")

# Row 3: Replay
st.subheader("📺 Validation Replay")
tab1, tab2 = st.tabs(["2D GIF Replay", "3D Interactive View"])

with tab1:
    if gifs and selected_gif:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(selected_gif, caption=os.path.basename(selected_gif), use_container_width=True)
    else:
        st.info("Waiting for first validation run...")

with tab2:
    htmls = sorted(glob.glob(os.path.join(checkpoint_dir, "*_3d.html")), key=os.path.getmtime, reverse=True)
    if htmls:
        selected_html = st.selectbox("Select 3D Visualization", htmls)
        with open(selected_html, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=800, scrolling=False)
    else:
        st.info("No 3D visualizations found yet.")