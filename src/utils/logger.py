import os
import csv
import time
import numpy as np

class FlightRecorder:
    """
    Logs detailed flight data for post-analysis.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.current_episode_data = []
        self.episode_count = 0
        self.headers = [
            "episode", "step", "time", 
            "blue_x", "blue_y", "blue_alt", "blue_heading", "blue_speed", "blue_g",
            "red_x", "red_y", "red_alt", "red_heading", "red_speed", "red_g",
            "action_roll", "action_g", "action_throttle", "action_fire", "action_cm",
            "reward", "is_locked", "missile_active"
        ]

    def log_step(self, episode, step, time_sec, blue_ent, red_ent, action, reward, is_locked, missile_active):
        """
        Buffer a single step of data.
        """
        # Handle missing entities (e.g. if dead)
        if blue_ent:
            b_x, b_y, b_alt = blue_ent.x, blue_ent.y, blue_ent.alt
            b_hdg, b_spd, b_g = blue_ent.heading, blue_ent.speed, blue_ent.g_load
        else:
            b_x, b_y, b_alt, b_hdg, b_spd, b_g = 0, 0, 0, 0, 0, 0

        if red_ent:
            r_x, r_y, r_alt = red_ent.x, red_ent.y, red_ent.alt
            r_hdg, r_spd, r_g = red_ent.heading, red_ent.speed, red_ent.g_load
        else:
            r_x, r_y, r_alt, r_hdg, r_spd, r_g = 0, 0, 0, 0, 0, 0

        row = [
            episode, step, time_sec,
            b_x, b_y, b_alt, b_hdg, b_spd, b_g,
            r_x, r_y, r_alt, r_hdg, r_spd, r_g,
            action[0], action[1], action[2], action[3], action[4],
            reward, int(is_locked), int(missile_active)
        ]
        self.current_episode_data.append(row)

    def save_episode(self, episode_id):
        """
        Write buffered data to CSV.
        """
        if not self.current_episode_data:
            return

        filename = os.path.join(self.log_dir, f"flight_record_ep{episode_id}_{int(time.time())}.csv")
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                writer.writerows(self.current_episode_data)
            # print(f"Flight record saved: {filename}")
        except Exception as e:
            print(f"Failed to save flight record: {e}")
        
        self.current_episode_data = []
