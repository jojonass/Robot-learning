import os
import torch
import metaworld
import random
import numpy as np
import gymnasium as gym
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# Ensure your utility file is named exactly 'utlitlies.py' based on your import
from utlitlies import * # ==================== CONFIGURATION ====================
ALGO_NAME = "SAC"
BENCHMARK_MODE = "MT3" 
EXPERIMENT_BATCH_ID = os.environ.get("BATCH_ID", "default_batch")

NUM_ENV = 24
TOTAL_TIMESTEPS = 5_000_000

# ==================== SCHEDULERS ====================
def cosine_annealing(initial_value, total_steps=TOTAL_TIMESTEPS, warmup_steps=3_000_000):
    def func(progress_remaining):
        current_step = (1 - progress_remaining) * total_steps
        if current_step < warmup_steps:
            return initial_value
        remaining_steps = total_steps - warmup_steps
        progress = (current_step - warmup_steps) / remaining_steps
        return initial_value * 0.5 * (1 + np.cos(np.pi * progress))
    return func

HYPERPARAMS = dict(
    learning_rate=3e-4, 
    buffer_size=2_000_000,
    learning_starts=40_000,
    batch_size=1024,
    tau=0.0005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=2,
    ent_coef='auto_0.1',
    target_entropy=-4.0,
    policy_kwargs=dict(
        net_arch=[512, 512, 512],
        activation_fn=torch.nn.ReLU,
        optimizer_kwargs=dict(eps=5e-5)
    )
)

# ==================== ENVIRONMENT WRAPPER ====================
class MT3Wrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.benchmark = metaworld.MT50()
        self.task_names = ['reach-v3', 'push-v3', 'pick-place-v3']
        self.all_tasks = [t for t in self.benchmark.train_tasks if t.env_name in self.task_names]

        dummy_env = self.benchmark.train_classes[self.task_names[0]]()
        self.base_obs_dim = dummy_env.observation_space.shape[0] # 39
        obs_shape = self.base_obs_dim + len(self.task_names) # 42

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = dummy_env.action_space
        self.active_env = None

    def _get_one_hot(self, task_idx):
        one_hot = np.zeros(len(self.task_names), dtype=np.float32)
        one_hot[task_idx] = 1.0
        return one_hot

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        task = random.choice(self.all_tasks)
        self.current_task_idx = self.task_names.index(task.env_name)
        env_cls = self.benchmark.train_classes[task.env_name]
        
        if self.active_env is None or not isinstance(self.active_env, env_cls):
            self.active_env = env_cls()
            
        self.active_env._freeze_rand_vec = False
        self.active_env.set_task(task)
        
        raw_obs, info = self.active_env.reset(seed=seed)
        info["task_name"] = self.task_names[self.current_task_idx]
        
        full_obs = np.concatenate([raw_obs, self._get_one_hot(self.current_task_idx)])
        return full_obs.astype(np.float32), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.active_env.step(action)
        task_name = self.task_names[self.current_task_idx]
        info["task_name"] = task_name
        
        full_obs = np.concatenate([raw_obs, self._get_one_hot(self.current_task_idx)])

        # Reward Shaping
        if info.get("success", 0) > 0:
            if "reach" in task_name: reward += 2.0
            elif "pick-place" in task_name: reward += 5000.0
            elif "push" in task_name: reward += 200.0
    
        reward = np.log1p(reward)
        return full_obs.astype(np.float32), reward, terminated, truncated, info

def make_env(rank, seed):
    def _init():
        set_random_seed(seed + rank)
        return Monitor(gym.wrappers.TimeLimit(MT3Wrapper(), max_episode_steps=500))
    return _init

# ==================== MAIN EXECUTION ====================#
if __name__ == "__main__":
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    SEED = int(slurm_id) if slurm_id else 42
    
    UNIQUE_RUN_ID = f"{ALGO_NAME}_{BENCHMARK_MODE}_seed{SEED}"
    BASE_LOG_DIR = f"./metaworld_logs/{ALGO_NAME}_{BENCHMARK_MODE}_{EXPERIMENT_BATCH_ID}"
    SEED_DIR = f"{BASE_LOG_DIR}/{UNIQUE_RUN_ID}" 
    os.makedirs(SEED_DIR, exist_ok=True)

    # 1. Environments
    env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENV)])
    env = SelectiveVecNormalize(env, norm_dim=39, norm_obs=True, norm_reward=False)
    
    eval_env = SubprocVecEnv([make_env(99, SEED)])
    eval_env = SelectiveVecNormalize(eval_env, norm_dim=39, norm_obs=True, training=False)
    eval_env.obs_rms = env.obs_rms

    # 2. Model
    model = SAC(policy="MlpPolicy", env=env, tensorboard_log=BASE_LOG_DIR, verbose=1, **HYPERPARAMS)

    # 3. --- CALLBACKS SETUP ---
    task_list = ['reach-v3', 'push-v3', 'pick-place-v3']
    
    success_tracker = EvalSuccessTracker(
        task_list=task_list, 
        seed=SEED
    )
    
    # Ensure this is an absolute path or a shared root so all seeds find it
    success_tracker.master_csv = os.path.abspath("combined_eval_success.csv")

    eval_callback = EvalCallbackMT3(
        eval_env,
        best_model_save_path=f"{SEED_DIR}/best_model/",
        eval_freq=max(1, 20000 // NUM_ENV),
        n_eval_episodes=15, 
        callback_after_eval=success_tracker, 
        verbose=1
    )

    # Manual link for safety
    success_tracker.parent = eval_callback

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 100000 // NUM_ENV),
        save_path=f"{SEED_DIR}/checkpoints/",
        name_prefix=UNIQUE_RUN_ID
    )

    # 4. --- TRAINING ---. 
   
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback], 
        tb_log_name=UNIQUE_RUN_ID,
        progress_bar=True
    )
    


    # 5. Final Save
    model.save(f"{SEED_DIR}/{UNIQUE_RUN_ID}_final")
    env.save(f"{SEED_DIR}/vecnormalize.pkl")
    
    env.close()
    eval_env.close()
