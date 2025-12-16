import os
if 'MUJOCO_GL' in os.environ:
    os.environ.pop('MUJOCO_GL', None)

import warnings
import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import  SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv



os.environ['MUJOCO_GL'] = 'glfw'


# ==================== DEVICE CONFIGURATION ====================
print("=" * 60)
if torch.cuda.is_available():
    TARGET_DEVICE = "cuda"
    print(f" CUDA available! Training will run on: {torch.cuda.get_device_name(0)}")
else:
    TARGET_DEVICE = "cpu"
    warnings.warn(" CUDA not available. Training will default to CPU.", UserWarning)
print("=" * 60)



def make_env(env_id, rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
    def _init():
        env = gym.make(env_id, seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


# ==================== CONFIGURATION ====================
# Task Selection
TASK_LIST = [
    'reach-v3',
    'push-v3',
    'pick-place-v3',
]
#A string for directory/model naming
TASK_NAME_STR = "MT3_" + "_".join([t.split('-')[0] for t in TASK_LIST])

# Define the unique ID for your custom MT3 environment (Used for gym.make)
MT3= f"Meta-World/{TASK_NAME_STR}-Custom-v3"
MAX_EPISODE_STEPS = 500 # Define this early for the register call


# 1. Get Task Classes and Map for Registration
try:
    all_envs = metaworld.ALL_V3_ENVIRONMENTS
    task_name_to_env_cls = {}
    for v3_name in TASK_LIST:
        key = v3_name
        if key not in all_envs:
            raise KeyError(f"Task key {key} not found in metaworld.ALL_V3_ENVIRONMENTS. Check if the task name is correct.")
        task_name_to_env_cls[v3_name] = all_envs[key]

except Exception as e:
    print(f"FATAL ERROR: Could not find all task definitions in ALL_V3_ENVIRONMENTS: {e}")
    print("Please verify your metaworld version and task names.")
    raise


# Use the official Meta-World entry point: 'metaworld.envs.multitask:MTEnv'
if MT3 not in gym.envs.registration.registry:
    gym.register(
        id=MT3,
        entry_point='metaworld.envs.multitask:MTEnv',
        kwargs={
            'task_name_to_env_cls': task_name_to_env_cls, # Use our mapped dictionary
            'max_episode_steps': MAX_EPISODE_STEPS,
            'auto_reset': True,
            'rand_init': True
        }
    )


if __name__ == "__main__":

    # Algorithm Selection
    ALGORITHM = "SAC"  # "TD3" or "DDPG" - SAC recommended for Meta-World
    # Environment Settings
    USE_PARALLEL = True  # Set to False for single environment
    N_ENVS = 8 if USE_PARALLEL else 1
    SEED = 42


    # Training Settings
    TOTAL_TIMESTEPS = 1_000_000  # Increased for better convergence
    MAX_EPISODE_STEPS = 500  # Maximum steps per episode
    NORMALIZE_REWARD = False  # Set to True if experiencing training instability
    # Evaluation Settings
    EVAL_FREQ = 10000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 25000  # Save checkpoint every N steps
    # ======================================================

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print(f"Meta-World MT3 Training: {MT3 }")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Create vectorized training environments (parallel)

    if USE_PARALLEL:
        print(f"Creating {N_ENVS} parallel training environments...")
        env = SubprocVecEnv(
            [make_env(MT3 , i, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD) for i in range(N_ENVS)],
            start_method='spawn'
        )
    else:
        print("Creating single training environment...")
        env = make_env(MT3, 0, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD)()


    # Create evaluation environment (without reward normalization for accurate eval)
    print("Creating evaluation environment...")
    eval_env = make_env(MT3, 0, SEED + 1000, MAX_EPISODE_STEPS, normalize_reward=False)()
    # Get action space dimensions
    n_actions = env.action_space.shape[0]
    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")



    if ALGORITHM == "SAC":
        # SAC - Recommended for Meta-World (better exploration)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5000,  # Start training sooner
            batch_size=256,
            tau=0.005,
            gamma=0.99,  # Higher gamma for multi-step tasks
            train_freq=1,
            gradient_steps=-1,  # Train on all available data
            ent_coef='auto',  # Automatic entropy tuning - crucial for SAC
            target_entropy='auto',  # Automatically set target entropy
            use_sde=False,  # State-dependent exploration (can be enabled for more exploration)
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # Deeper network
                activation_fn=torch.nn.ReLU,
                log_std_init=-3,  # Initial exploration level
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}_{TASK_NAME_STR}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Callbacks
    # Save checkpoint every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{MT3}/",
        name_prefix=f"{ALGORITHM.lower()}_{MT3}",
        verbose=1
    )

    # Evaluate every EVAL_FREQ steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{MT3}/",
        log_path=f"./metaworld_logs/eval_{MT3}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,  # More episodes for robust evaluation
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    # Train the agent
    total_timesteps = TOTAL_TIMESTEPS
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    print("Training configuration:")
    print(f"  - Task: {MT3}")
    print(f"  - Algorithm: {ALGORITHM}")
    print(f"  - Parallel environments: {N_ENVS}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma: {model.gamma}")
    print(f"  - Learning starts: {model.learning_starts}")
    print(f"  - Buffer size: {model.buffer_size:,}")
    print(f"  - Network architecture: [256, 256, 256]")
    print(f"  - Gradient steps: -1 (train on all data)")
    print(f"  - Seed: {SEED}")
    print(f"  - Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  - Reward function: v2 (more stable)")
    print(f"  - Normalize reward: {NORMALIZE_REWARD}")
    print(f"  - Eval frequency: {EVAL_FREQ} steps")
    print(f"  - Eval episodes: {N_EVAL_EPISODES}")
    print(f"  - Checkpoint frequency: {CHECKPOINT_FREQ} steps")
    if ALGORITHM == "TD3":
        print(f"  - Exploration noise: σ=0.1")
        print(f"  - Target policy noise: 0.1 (clip: 0.3)")
    elif ALGORITHM == "SAC":
        print(f"  - Entropy tuning: Automatic")
        print(f"  - Target entropy: Automatic")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_{MT3}_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_{MT3}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{MT3}/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{MT3}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()
    eval_env.close()


    #  to find training curves run tensorboard --logdir=./metaworld_logs/ in terminal
