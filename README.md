# Multi-Task SAC: Meta-World Benchmark (Group 8)

This repository contains the implementation of a custom Soft Actor-Critic (SAC) agent designed for the Meta-World benchmark. The project scales reinforcement learning from single-task baselines (MT1) to multi-task proficiency (MT3 and MT10) using task conditioning and hierarchical reward shaping.

##  Repository Structure

### Training Scripts
* **`train_metaworld_MT1_reach.py`**: Baseline training for the Reach task.
* **`train_metaworld_MT1_push.py`**: Training script for the Push task with specialized rewards.
* **`train_Metaworld_MT1_pick.py`**: Training script for the Pick-and-Place task.
* **`train_metaworld_MT3.py`**: Multi-task training for the 3-task suite (Reach, Push, Pick-place).
* **`train_metaworld_MT10.py`**: Large-scale training for the 10-task benchmark using a [768, 768, 768] MLP.

### Evaluation & Utilities
* **`eval_MT1.py` / `eval_MT.py`**: Scripts for validating models and generating success statistics.
* **`play_metaworld_sb3.py`**: Script for visualizing agent behavior and rendering performance.
* **`CSV.py`**: Script used to process the training logs and generate plots from CSV data.
* **`utlitlies`**: Directory containing environment wrappers and reward transformation logic.

##  Logging & Metrics in Data folder
The training process generates CSV logs for every task and stability metric. Key files found in the logs include:
* **`eval_success_mean_combined`**: The aggregate success rate across the task suite.
* **`train_ent_coef` / `train_ent_coef_loss`**: Data regarding entropy alpha and its optimization.
* **`train_actor_loss` / `train_critic_loss`**: Training dynamics of the SAC networks.
* **`eval_success_[task_name]-v3`**: Individual success rates for specific tasks like Peg-Insert, Window-Open, and Drawer-Close.

##  Setup & Requirements

### Environment
* **Python**: 3.10
* **RL Framework**: Gymnasium and Meta-World (v2/v3 tasks)
* **Deep Learning**: PyTorch
* **Hardware**: Training conducted on the TU Wien high-performance computing cluster

