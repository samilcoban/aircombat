# ================================================
# FILE: src/utils/logger.py
# ================================================
import os
import csv
import time
import math
import numpy as np


class FlightRecorder:
    """
    Compact Flight Recorder.
    Logs essential telemetry for analysis.

    UPDATED: Handles Unit Conversion (Radians -> Degrees, m/s -> Knots)
    so logs remain human-readable despite internal physics changes.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.buffer = []
        self.buffer_size = 100  # Write to disk every 100 steps
        self.episode_id = 0

        self.headers = [
            "ep", "step", "t",
            "id", "team",
            "x", "y", "alt",
            "hdg_deg", "spd_kts", "g",
            "roll_cmd", "g_cmd", "thr_cmd", "fire", "cm",
            "rew"
        ]

        self.current_file = None
        self.writer = None

    def start_episode(self, episode_id):
        """Starts a new log file for a new episode."""
        self.flush()
        self.episode_id = episode_id
        timestamp = int(time.time())
        filename = os.path.join(self.log_dir, f"flight_log_{episode_id}_{timestamp}.csv")

        try:
            self.current_file = open(filename, 'w', newline='')
            self.writer = csv.writer(self.current_file)
            self.writer.writerow(self.headers)
        except Exception as e:
            print(f"Logger Error: {e}")

    def log_step(self, agent_id, team, step, time_sec, ent, action, reward):
        """Logs a single step of telemetry for an agent."""
        if not self.current_file: return
        if not ent: return

        # CONVERSION: Radians -> Degrees for CSV readability
        hdg_deg = math.degrees(ent.heading) % 360.0

        # CONVERSION: m/s -> Knots
        # Internal speed is knots? Check core_flat.py.
        # "speed: float = 0.0 # Knots" -> Entity def says Knots.
        # Physics update says: speed_ms = ent.speed * KNOTS_TO_MS.
        # So ent.speed IS stored in Knots. No conversion needed for speed.
        spd_kts = ent.speed

        row = [
            self.episode_id, step, f"{time_sec:.2f}",
            agent_id, team,
            f"{ent.x:.1f}", f"{ent.y:.1f}", f"{ent.alt:.1f}",
            f"{hdg_deg:.1f}", f"{spd_kts:.1f}", f"{ent.g_load:.2f}",
            f"{action[0]:.2f}", f"{action[1]:.2f}", f"{action[2]:.2f}",
            f"{action[3]:.1f}", f"{action[4]:.1f}",
            f"{reward:.4f}"
        ]

        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Writes buffered data to disk."""
        if self.current_file and self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer = []
            self.current_file.flush()

    def close(self):
        self.flush()
        if self.current_file:
            self.current_file.close()
            self.current_file = None