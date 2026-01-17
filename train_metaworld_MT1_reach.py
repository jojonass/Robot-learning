import os
import metaworld
import gymnasium as gym
import torch
import numpy as np
import random
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# ==================== SETTINGS & HYPERPARAMS ==================== 
ALGO_NAME = "SAC"
TASK_NAME = 'reach-v3'

EXPERIMENT_BATCH_ID = os.environ.get("BATCH_ID", "debug")
# This ID can be your "Batch Name" (e.g., "Default_Params" or "Experiment_A")


NUM_ENV = 8
TOTAL_TIMESTEPS = 330_000

HYPERPARAMS = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=40_000,
    batch_size=1024,
    tau=0.001,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto_4.0',
    target_entropy=-3.0,
    policy_kwargs=dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU 
    )
)

# For slurm / consistency
def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def make_env(task_name, env_id, base_seed):
    def _init():
        seed = base_seed + env_id
        mt1 = metaworld.MT1(task_name)
        env = mt1.train_classes[task_name]()
        env = SuccessFixer(env, mt1.train_tasks, seed)
        env = Monitor(env) 
        return env
    return _init

class SuccessFixer(gym.Wrapper):
    def __init__(self, env, tasks, seed=None):
        super().__init__(env)
        self.tasks = tasks
        self.was_successful = 0
        self.np_random = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # FIX: Pick an integer index instead of the object directly
        num_tasks = len(self.tasks)
        idx = self.np_random.integers(0, num_tasks)
        task = self.tasks[idx]
        
        self.env.set_task(task)
        self.was_successful = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", 0) > 0:
            self.was_successful = 1
        info["is_success"] = self.was_successful
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Get Slurm ID or default to 1
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    SEED = int(slurm_task_id) if slurm_task_id else 1
    set_global_seeds(SEED)
    
        
    FULL_EXP_ID = f"{ALGO_NAME}_{TASK_NAME}_{EXPERIMENT_BATCH_ID}"
    UNIQUE_RUN_ID = f"{ALGO_NAME}_{TASK_NAME}_seed{SEED}"
    BASE_LOG_DIR = f"./metaworld_logs/{FULL_EXP_ID}"
    SEED_DIR = f"{BASE_LOG_DIR}/{UNIQUE_RUN_ID}" 
    
    os.makedirs(f"./results/{FULL_EXP_ID}/hyperparameters", exist_ok=True)
    os.makedirs(f"{SEED_DIR}/best_model", exist_ok=True)
    os.makedirs(f"{SEED_DIR}/checkpoints", exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    # Save Config
    config_path = f"./results/{FULL_EXP_ID}/hyperparameters/config.txt"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(f"Experiment: {FULL_EXP_ID}\nHyperparams: {str(HYPERPARAMS)}\n")

    # --- SETUP & TRAINING ---
    env = SubprocVecEnv([make_env(TASK_NAME, i, SEED) for i in range(NUM_ENV)])
    eval_env = SubprocVecEnv([make_env(TASK_NAME, i + 100, SEED) for i in range(NUM_ENV)])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=BASE_LOG_DIR,
        verbose=1,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **HYPERPARAMS 
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // NUM_ENV, 1),
        save_path=f"{SEED_DIR}/checkpoints/",
        name_prefix=f"{ALGO_NAME}_{TASK_NAME}"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{SEED_DIR}/best_model/",
        log_path=f"{BASE_LOG_DIR}/{UNIQUE_RUN_ID}/eval/",
        eval_freq=max(5000 // NUM_ENV, 1),
        n_eval_episodes=20,
        deterministic=True
    )

    print(f" Running: {UNIQUE_RUN_ID}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=UNIQUE_RUN_ID
    )

    model.save(f"{SEED_DIR}/{UNIQUE_RUN_ID}_final")
    env.close()
    eval_env.close()
