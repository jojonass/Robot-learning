import os
os.environ["MUJOCO_GL"] = "glfw"
import torch
import metaworld
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from utlitlies import *


# ==================== DEVICE CONFIGURATION ====================
print("=" * 60)
if torch.cuda.is_available():
    TARGET_DEVICE = "cuda"
    #print(f" CUDA available! Training will run on: {torch.cuda.get_device_name(0)}")
else:
    TARGET_DEVICE = "cpu"
    #warnings.warn(" CUDA not available. Training will default to CPU.", UserWarning)
print("=" * 60)


class MT10Wrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.benchmark = metaworld.MT10()
        self.all_tasks = self.benchmark.train_tasks

        # 1. Map task names to IDs (0-9)
        self.task_names = list(self.benchmark.train_classes.keys())

        # 2. Define spaces
        # MT10 obs is 39-dim. We add 10-dim for One-Hot = 49-dim total.
        dummy_env = self.benchmark.train_classes['reach-v3']()
        obs_shape = dummy_env.observation_space.shape[0] + 10

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        self.action_space = dummy_env.action_space
        self.active_env = None
        self.current_task_idx = 0

    def _get_one_hot(self, task_idx):
        one_hot = np.zeros(10, dtype=np.float32)
        one_hot[task_idx] = 1.0
        return one_hot

    def reset(self, seed=None, options=None):
        # Pick random task
        task = random.choice(self.all_tasks)
        self.current_task_idx = self.task_names.index(task.env_name)

        # Create env for that specific task
        env_cls = self.benchmark.train_classes[task.env_name]
        if self.active_env is None or self.active_env.__class__ != env_cls:
            self.active_env = env_cls()
        self.active_env._freeze_rand_vec = False  # Ensure goal randomization
        self.active_env.set_task(task)

        # Reset and get base obs
        raw_obs = self.active_env.reset(seed=seed)
        if isinstance(raw_obs, tuple): raw_obs = raw_obs[0]  # Handle Gym API versions

        # Append One-Hot Task ID
        full_obs = np.concatenate([raw_obs, self._get_one_hot(self.current_task_idx)])
        return full_obs.astype(np.float32), {}

    def step(self, action):

        raw_obs, reward, terminated, truncated, info = self.active_env.step(action)
        # Append One-Hot to step observation too!
        full_obs = np.concatenate([raw_obs, self._get_one_hot(self.current_task_idx)])
        info["task_name"] = self.task_names[self.current_task_idx]
        return full_obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        return self.active_env.render()

    def close(self):
        if self.active_env is not None:
            self.active_env.close()


if __name__ == "__main__":


    TOTAL_TIMESTEPS = 50000  # Increased for better convergence
    MAX_EPISODE_STEPS = 500  # Maximum steps per episode
    ALGORITHM = "SAC"
    env = gym.wrappers.TimeLimit(MT10Wrapper(), max_episode_steps=MAX_EPISODE_STEPS)
    eval_env = gym.wrappers.TimeLimit(MT10Wrapper(), max_episode_steps=MAX_EPISODE_STEPS)
    success_callback = SuccessCallback(log_freq=10_000)
    SEED = 42


    # Evaluation Settings
    EVAL_FREQ = 10000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 25000  # Save checkpoint every N steps

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print("Meta-World MT10 Training")
    print(f"=" * 60)



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
            tensorboard_log="./metaworld_logs/MT10/",
            verbose=1,
            device="auto",
            seed=SEED,
        )

    # Save checkpoint every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_MT10/",
        name_prefix=f"{ALGORITHM.lower()}_MT10",
        verbose=1
    )

    # Evaluate every EVAL_FREQ steps
    eval_callback =  EvalSuccessCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_MT10/",
        log_path=f"./metaworld_logs/eval_MT10/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,  # More episodes for robust evaluation
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    print("MT10 Task-Conditioned Wrapper initialized.")
    print(f"Observation space: {env.observation_space.shape}")  # Should be (49,)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback,success_callback],
        log_interval=10,
        progress_bar=True
    )
    print("MT10 Task-Conditioned Wrapper learn success.")

    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/_MT10_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/_MT10_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_MT10/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_MT10/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()
