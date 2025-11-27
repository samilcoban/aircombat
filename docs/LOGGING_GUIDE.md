# Enhanced Logging System Guide

## Overview

The training system now includes comprehensive logging to track agent progress despite reward normalization wrappers.

## TensorBoard Metrics

### Charts (Training Progress)
- **`charts/loss`**: PPO loss (policy + value + entropy)
- **`charts/mean_step_reward_normalized`**: Normalized rewards (will be near 0, expected)
- **`charts/learning_rate`**: Current learning rate

### Episode Statistics (Raw, Pre-Normalization)
These metrics show **actual agent performance** before normalization:
- **`episode/raw_return_mean`**: Average episode return (THIS IS YOUR MAIN METRIC)
- **`episode/raw_return_max`**: Best episode return
- **`episode/raw_return_min`**: Worst episode return
- **`episode/length_mean`**: Average episode length in steps
- **`episode/count`**: Number of episodes completed in this update

### Termination Reasons
Track how episodes end:
- **`termination/crash`**: Agent crashed into ground
- **`termination/shot`**: Agent was shot down
- **`termination/win`**: Agent won (killed all enemies)
- **`termination/timeout`**: Episode reached time limit
- **`termination/floor_violation`**: Altitude < 2000m

### Actions (Behavior Monitoring)
Monitor agent's action distribution:
- **`actions/roll_mean`** & **`actions/roll_std`**: Roll input statistics
- **`actions/g_pull_mean`** & **`actions/g_pull_std`**: **KEY METRIC** - should decrease from 0.6 to 0.2-0.4
- **`actions/throttle_mean`** & **`actions/throttle_std`**: Throttle control
- **`actions/fire_mean`** & **`actions/fire_std`**: Firing behavior

### Curriculum Tracking
Monitor curriculum learning progress:
- **`curriculum/phase`**: Current phase (1-4)
- **`curriculum/phase_progress`**: Progress within phase (0.0-1.0)
- **`curriculum/kappa`**: Opponent difficulty (1.0=easy, 0.0=expert)

### Value Function
Monitor critic learning:
- **`value/mean`**: Average value prediction
- **`value/std`**: Value prediction variance
- **`value/max`**: Maximum value prediction

## Checkpoint Inspection Tool

### Usage

```bash
# Inspect latest checkpoint
python scripts/inspect_checkpoint.py checkpoints/model_latest.pt

# Inspect specific checkpoint with 5 episodes
python scripts/inspect_checkpoint.py checkpoints/model_100.pt --episodes 5

# Custom output directory
python scripts/inspect_checkpoint.py checkpoints/model_200.pt --output-dir my_logs
```

### Output

Creates detailed text log in `logs/inspections/` with:
- **Initial State**: Spawn positions, separation distance
- **Step-by-Step Logs** (every 50 steps):
  - Actions (all 5 dimensions)
  - Reward (step and cumulative)
  - Value prediction
  - Physics state (G-load, speed, altitude, fuel)
  - Distance to nearest enemy
- **Episode Summary**:
  - Outcome (crash/shot/win/timeout)
  - Total steps and reward
  - Action statistics (mean, std, min, max)
  - Physics statistics (G-load, speed, altitude)
  - G-load distribution histogram
- **Overall Summary** (across all episodes):
  - Average reward, steps, G-load
  - Outcome distribution

### Example Output Snippet

```
=== Step 0 ===
  Action: roll=+0.123, g_pull=-0.456, throttle=+0.789, fire=-0.012, cm=+0.034
  Reward: +0.0050 (cumulative: +0.0050)
  Value: 0.1234
  Physics: G=1.23, speed=600km/h, alt=5000m, fuel=1.00
  Nearest enemy: 8.45 km

...

EPISODE 1 SUMMARY
Outcome: timeout
Steps: 1200
Total Reward: +12.3456

Action Statistics:
  G-Pull:   mean=+0.234, std=0.123, min=-0.567, max=+0.890

Physics Statistics:
  G-Load:   mean=2.34, max=5.67, min=1.00
  
  G-Load Distribution:
    0.0-1.5G:  45.2% (542 steps)
    1.5-3.0G:  32.1% (385 steps)
    3.0-6.0G:  20.5% (246 steps)
    6.0-9.0G:   2.2% (27 steps)
```

## Key Metrics to Watch

### For Spiraling Fix
1. **`actions/g_pull_mean`**: Should decrease from ~0.6 to 0.2-0.4 range
2. **G-Load Distribution** (in inspection logs): Should show more time in 0-3G range
3. **`episode/raw_return_mean`**: Should increase over time (agent learning)

### For Curriculum Progress
1. **`curriculum/phase`**: Tracks which phase you're in
2. **`episode/raw_return_mean`**: Should stay positive and increase
3. **`termination/crash`** vs **`termination/timeout`**: Fewer crashes = progress

### For Overall Learning
1. **`episode/raw_return_mean`**: Main metric - should trend upward
2. **`episode/length_mean`**: Longer episodes = agent surviving longer
3. **`value/mean`**: Should stabilize (not explode or collapse)

## Recommended TensorBoard Layout

Create custom scalars in TensorBoard:
1. **Training Overview**: `episode/raw_return_mean`, `episode/length_mean`, `charts/loss`
2. **Behavior**: `actions/g_pull_mean`, `actions/throttle_mean`, `actions/fire_mean`
3. **Curriculum**: `curriculum/phase`, `curriculum/kappa`, `episode/raw_return_mean`
4. **Terminations**: All `termination/*` metrics
