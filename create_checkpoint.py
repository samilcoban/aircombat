
import torch
from src.model import AgentTransformer
from config import Config
import os

def create_checkpoint():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        
    print("Creating dummy checkpoint...")
    model = AgentTransformer().to(Config.DEVICE)
    
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'update': 0
    }
    
    torch.save(checkpoint_data, "checkpoints/model_latest.pt")
    print("Saved checkpoints/model_latest.pt")

if __name__ == "__main__":
    create_checkpoint()
