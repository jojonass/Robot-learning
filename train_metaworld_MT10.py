import os
if 'MUJOCO_GL' in os.environ:
    os.environ.pop('MUJOCO_GL', None)
import warnings
import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import TD3, DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import warnings
from stable_baselines3.common.env_util import make_vec_env
import random
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor



os.environ['MUJOCO_GL'] = 'glfw'
os.environ['GYM_WARN'] = '0'


# ==================== DEVICE CONFIGURATION ====================
if torch.cuda.is_available():
    TARGET_DEVICE = "cuda"
    #print(f" CUDA available! Training will run on: {torch.cuda.get_device_name(0)}")
else:
    TARGET_DEVICE = "cpu"
    #warnings.warn(" CUDA not available. Training will default to CPU.", UserWarning)


class MetaWorldMT10Env(gym.Env):
    def __init__(self, seed=0):
        self.mt = metaworld.MT10(seed=seed)
        self.tasks = self.mt.train_tasks
        self.env_classes = self.mt.train_classes

        # Use one env to define spaces
        sample_env = list(self.env_classes.values())[0]()
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space

        self.env = None

    def reset(self):
        # Sample a task
        task = random.choice(self.tasks)
        env_cls = self.env_classes[task.env_name]

        self.env = env_cls()
        self.env.set_task(task)

        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

print("MT10 wrapper works!")

def make_env():
    return MetaWorldMT10Env(seed=42)

env = DummyVecEnv([make_env])
env = VecMonitor(env)


obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

"""
if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Task Selection
    TASK_NAME = 'MT10-v3'

    # Algorithm Selection
    ALGORITHM = "SAC"  # "TD3" or "DDPG" - SAC recommended for Meta-World

    # Environment Settings
    N_ENVS = 10  # 10 enviroments are made for MT10
    SEED = 42

    # Training Settings
    TOTAL_TIMESTEPS = 50000  # Increased for better convergence
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
    print(f"Meta-World MT10 Training: {TASK_NAME}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Create the training environment
    print("Creating MT10 multi-task training environment...")
    # SubprocVecEnv requires a list of callable functions (your make_env returns _init)
    env = SubprocVecEnv(
        [make_env(TASK_NAME, i, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD) for i in range(N_ENVS)],
        start_method='spawn'
    )

    # Create evaluation environment (can skip SubprocVecEnv for eval for simplicity)
    print("Creating evaluation environment...")
    eval_env = make_env(TASK_NAME, 0, SEED + 1000, MAX_EPISODE_STEPS, normalize_reward=False)()
    # The eval env will still be SyncVectorEnv, but EvalCallback handles it better than the model init




    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")


    if ALGORITHM == "SAC":
        # SAC - Recommended for Meta-World (better exploration)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100_000,
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
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
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
        save_path=f"./metaworld_models/checkpoints_{TASK_NAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{TASK_NAME}",
        verbose=1
    )

    # Evaluate every EVAL_FREQ steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{TASK_NAME}/",
        log_path=f"./metaworld_logs/eval_{TASK_NAME}/",
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
    print(f"  - Task: {TASK_NAME}")
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
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{TASK_NAME}/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{TASK_NAME}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()
    eval_env.close()


    #  to find training curves run tensorboard --logdir=./metaworld_logs/ in terminal
"""