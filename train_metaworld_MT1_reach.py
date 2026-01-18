import os
import torch
import metaworld
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from utlitlies import * # ==================== CONFIGURATION ====================
ALGO_NAME = "SAC"
BENCHMARK_MODE = "MT10" 
EXPERIMENT_BATCH_ID = os.environ.get("BATCH_ID", "MT10_custom_shaping")

NUM_ENV = 40
TOTAL_TIMESTEPS = 8_000_000 

# ==================== SCHEDULERS ====================
def cosine_annealing(initial_value, total_steps=TOTAL_TIMESTEPS, warmup_steps=7_000_000):
    def func(progress_remaining):
        current_step = (1 - progress_remaining) * total_steps
        if current_step < warmup_steps: return initial_value
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return initial_value * 0.5 * (1 + np.cos(np.pi * progress))
    return func

HYPERPARAMS = dict(
    learning_rate= 3e-4,  
    buffer_size=1_000_000,
    learning_starts=50_000,
    batch_size= 1024,
    tau=0.0005,
    gamma=0.995,
    train_freq=1,
    gradient_steps=2,
    ent_coef='auto_0.1',
    target_entropy=-4.0,
    policy_kwargs=dict(
        net_arch=[768, 768, 768],
        activation_fn=torch.nn.ReLU,
        optimizer_kwargs=dict(eps=5e-5)
    )
)

# ==================== ENVIRONMENT WRAPPER ====================
class MT10Wrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.benchmark = metaworld.MT10()
        self.task_names = list(self.benchmark.train_classes.keys())
        # Store tasks as a list to access by index
        self.all_tasks = self.benchmark.train_tasks 
        self.np_random = np.random.default_rng()
        
        # Pre-instantiate environments locally in each worker
        self.envs = {name: cls() for name, cls in self.benchmark.train_classes.items()}
        for e in self.envs.values():
            e._freeze_rand_vec = False

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)
        self.action_space = self.envs[self.task_names[0]].action_space

    def reset(self, seed=None, options=None):
        # Handle seeding properly for reproducibility in Subproc
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        task_idx = self.np_random.integers(0, len(self.all_tasks))
        task = self.all_tasks[task_idx]
        
        self.active_name = task.env_name
        self.current_task_idx = self.task_names.index(self.active_name)
        
        env = self.envs[self.active_name]
        env.set_task(task)
        
        raw_obs, info = env.reset(seed=seed)
        info["task_name"] = self.active_name
        
        one_hot = np.zeros(10, dtype=np.float32)
        one_hot[self.current_task_idx] = 1.0
        return np.concatenate([raw_obs, one_hot]).astype(np.float32), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.envs[self.active_name].step(action)
        info["task_name"] = self.active_name
        
        # Apply your working Reward Shaping Logic for MT10
        if info.get("success", 0) > 0:
            name = self.active_name.lower()
            if "pick-place" in name or "peg-insert-side" in name or "push" in name:
                reward += 20_000.0
            else: 
                reward += 200.0
    
        reward = np.log1p(np.clip(reward, -0.99, None))
        
        one_hot = np.zeros(10, dtype=np.float32)
        one_hot[self.current_task_idx] = 1.0
        full_obs = np.concatenate([raw_obs, one_hot])
        
        return full_obs.astype(np.float32), float(reward), terminated, truncated, info

def make_env(rank, seed):
    def _init():
        set_random_seed(seed + rank)
        return Monitor(gym.wrappers.TimeLimit(MT10Wrapper(), max_episode_steps=500))
    return _init

# ==================== MAIN ====================
if __name__ == "__main__":
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    SEED = int(slurm_id) if slurm_id else 42
    
    UNIQUE_RUN_ID = f"{ALGO_NAME}_{BENCHMARK_MODE}_seed{SEED}"
    BASE_LOG_DIR = f"./metaworld_logs/{ALGO_NAME}_{BENCHMARK_MODE}_{EXPERIMENT_BATCH_ID}"
    SEED_DIR = f"{BASE_LOG_DIR}/{UNIQUE_RUN_ID}" 
    os.makedirs(SEED_DIR, exist_ok=True)

    # 1. Training Environment
    env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENV)])
    env = SelectiveVecNormalize(env, norm_dim=39, norm_obs=True, norm_reward=False)
    
    # 2. Evaluation Environment
    eval_env = SubprocVecEnv([make_env(99, SEED)])
    eval_env = SelectiveVecNormalize(eval_env, norm_dim=39, norm_obs=True, training=False)
    eval_env.obs_rms = env.obs_rms

    # 3. Model
    model = SAC(policy="MlpPolicy", env=env, tensorboard_log=BASE_LOG_DIR, verbose=1, **HYPERPARAMS)

    # 4. Success Tracker & Callback
    dummy_mt10 = metaworld.MT10()
    mt10_tasks = list(dummy_mt10.train_classes.keys())
    
    success_tracker = EvalSuccessTracker(task_list=mt10_tasks, seed=SEED)
    success_tracker.master_csv = os.path.abspath("combined_mt10_success.csv")

    eval_callback = EvalCallbackMT3(
        eval_env,
        best_model_save_path=f"{SEED_DIR}/best_model/",
        eval_freq=max(1, 40000 // NUM_ENV),
        n_eval_episodes=50, # 5 per task
        callback_after_eval=success_tracker, 
        verbose=1
    )
    success_tracker.parent = eval_callback

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 200000 // NUM_ENV),
        save_path=f"{SEED_DIR}/checkpoints/",
        name_prefix=UNIQUE_RUN_ID
    )

    # 5. Training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback], 
        tb_log_name=UNIQUE_RUN_ID,
        progress_bar=True
    )

    model.save(f"{SEED_DIR}/{UNIQUE_RUN_ID}_final")
    env.save(f"{SEED_DIR}/vecnormalize.pkl")
    env.close()
    eval_env.close()
