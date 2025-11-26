#!/usr/bin/env python3
"""
Fix checkpoint files that were saved with torch.compile() wrapper.
Removes the '_orig_mod.' prefix from all state dict keys.
"""

import torch
import sys
import os

def fix_checkpoint(checkpoint_path):
    """
    Load a checkpoint and remove '_orig_mod.' prefix from state dict keys.
    
    Args:
        checkpoint_path: Path to the checkpoint file to fix
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle both dict and raw state_dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Check if any keys have the prefix
    has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    
    if not has_prefix:
        print("✅ Checkpoint is already clean (no _orig_mod. prefix found)")
        return True
    
    print("🔧 Removing _orig_mod. prefix from state dict keys...")
    
    # Remove the prefix
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Update the checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = cleaned_state_dict
    else:
        checkpoint = cleaned_state_dict
    
    # Save the fixed checkpoint
    backup_path = checkpoint_path + ".backup"
    print(f"📦 Creating backup: {backup_path}")
    os.rename(checkpoint_path, backup_path)
    
    print(f"💾 Saving fixed checkpoint: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    print("✅ Checkpoint fixed successfully!")
    print(f"   Original backed up to: {backup_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_checkpoint.py <checkpoint_path>")
        print("Example: python fix_checkpoint.py checkpoints/model_latest.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = fix_checkpoint(checkpoint_path)
    sys.exit(0 if success else 1)
