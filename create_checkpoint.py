# ================================================
# FILE: create_checkpoint.py
# ================================================
import torch
from src.model import HybridActorCritic  # <--- UPDATED
from config import Config
import os


def create_checkpoint():
    # Intuition: Ensure the directory for saving checkpoints exists.
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    print("Creating dummy checkpoint...")
    # Intuition: Initialize the new Hybrid model structure.
    # Math: Creates the neural network graph with random initial weights.
    model = HybridActorCritic().to(Config.DEVICE)

    # Intuition: Prepare the data dictionary to be saved.
    # Math: Stores the state dictionary (weights and biases) of the model.
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {}, # Intuition: Empty optimizer state for a fresh start.
        'update': 0 # Intuition: Initial update step count.
    }

    # Intuition: Serialize and save the checkpoint to disk.
    torch.save(checkpoint_data, "checkpoints/model_latest.pt")
    print("Saved checkpoints/model_latest.pt")


if __name__ == "__main__":
    create_checkpoint()