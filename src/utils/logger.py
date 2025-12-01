# ================================================
# FILE: src/utils/logger.py
# ================================================
import os
import csv
import time
import numpy as np


class FlightRecorder:
    """
    Compact Flight Recorder.
    Logs essential telemetry with reduced precision to save space.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.buffer = []
        self.buffer_size = 100  # Write every 100 steps
        self.episode_id = 0

        self.headers = [
            "ep", "step", "t",
            "id", "team", "x", "y", "alt", "hdg", "spd", "g",
            "roll_cmd", "g_cmd", "thr_cmd", "fire", "cm",
            "rew"
        ]

        self.current_file = None
        self.writer = None

    def start_episode(self, episode_id):
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
        if not self.current_file: return
        if not ent: return

        row = [
            self.episode_id, step, f"{time_sec:.1f}",
            agent_id, team,
            f"{ent.x:.0f}", f"{ent.y:.0f}", f"{ent.alt:.0f}",
            f"{ent.heading:.1f}", f"{ent.speed:.0f}", f"{ent.g_load:.2f}",
            f"{action[0]:.2f}", f"{action[1]:.2f}", f"{action[2]:.2f}",
            f"{action[3]:.1f}", f"{action[4]:.1f}",
            f"{reward:.3f}"
        ]

        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.current_file and self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer = []
            self.current_file.flush()

    def close(self):
        self.flush()
        if self.current_file:
            self.current_file.close()
            self.current_file = None


class SystemMonitor:
    """
    Monitors Hardware (GPU/CPU/RAM) stats for TensorBoard.
    Fails gracefully if libraries are missing.
    """

    def __init__(self):
        self.pynvml = None
        self.psutil = None
        self.handle = None

        # Try Initialize NVIDIA Management Library
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Default GPU 0
            # print("✅ SystemMonitor: NVIDIA GPU Detected")
        except ImportError:
            print("⚠️ SystemMonitor: 'nvidia-ml-py3' not installed. GPU logging disabled.")
        except Exception as e:
            print(f"⚠️ SystemMonitor: GPU Init failed: {e}")

        # Try Initialize PSUTIL
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            print("⚠️ SystemMonitor: 'psutil' not installed. CPU logging disabled.")

    def get_stats(self):
        stats = {}

        # GPU Stats
        if self.pynvml and self.handle:
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(self.handle, 0)  # 0 = GPU sensor
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                stats['hw/gpu_util'] = util.gpu
                stats['hw/gpu_mem_used_mb'] = mem.used / 1024 / 1024
                stats['hw/gpu_temp_c'] = temp
            except:
                pass

        # CPU/RAM Stats
        if self.psutil:
            try:
                stats['hw/cpu_util'] = self.psutil.cpu_percent()
                stats['hw/ram_util'] = self.psutil.virtual_memory().percent
            except:
                pass

        return stats