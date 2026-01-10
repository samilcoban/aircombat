# AirCombat 3.0 Documentation

## 📚 Overview
AirCombat 3.0 is a hierarchical multi-agent reinforcement learning framework designed for high-fidelity BVR (Beyond Visual Range) and WVR (Within Visual Range) combat.

This documentation is split into three technical pillars:

1.  [**Environment & Physics**](./docs/environment.md)
    *   State Space (Nodes/Edges), Action Space, and Physics Engine mechanics.
    *   Explanation of the Energy-Maneuverability model.
    *   Reward function details.

2.  [**Model Architecture**](./docs/models.md)
    *   The "Decoupled Commander" Architecture.
    *   Hybrid Transformer-GRU Actor.
    *   Global Graph Neural Network (GNN) Critic.
    *   Tiny Recursive Model (TRM) Heads.

3.  [**Training Process**](./docs/training.md)
    *   Curriculum Phases (School -> Combat -> War).
    *   Prioritized Fictitious Self-Play (PFSP).
    *   GAIL (Generative Adversarial Imitation Learning) pipeline.

## 🛠️ Quick Reference
*   **Config**: `config.py` contains all hyperparameters and physics constants.
*   **Entry Point**: `train.py` handles the PPO loop.
*   **Vis**: `dashboard.py` allows inspection of training runs.