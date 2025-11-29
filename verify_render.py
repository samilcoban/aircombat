
import torch
from src.model import AgentTransformer
from train import save_validation_gif, make_env
from config import Config

def verify():
    print("Initializing model...")
    model = AgentTransformer().to(Config.DEVICE)
    model.eval()
    
    print("Running save_validation_gif...")
    # Use update number 999 for verification
    save_validation_gif(model, step=999)
    
    print("Verification complete.")

if __name__ == "__main__":
    verify()
