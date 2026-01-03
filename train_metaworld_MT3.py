import os
os.environ["MUJOCO_GL"] = "egl"
import torch
import metaworld
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from utlitlies import * # Ensure this contains SuccessCallback and EvalSuccessCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import argparse

# ==================== DEVICE CONFIGURATION ====================
print("=" * 60)
if torch.cuda.is_available():
    TARGET_DEVICE = "cuda"
    print(f" CUDA available! Training will run on: {torch.cuda.get_device_name(0)}")
else:
    TARGET_DEVICE = "cpu"
    print(" CUDA not available. Training will default to CPU.")
print("=" * 60)

class MT3Wrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.benchmark = metaworld.MT50() 
        self.task_names = ['reach-v3', 'push-v3', 'pick-place-v3']
        self.all_tasks = [t for t in self.benchmark.train_tasks if t.env_name in self.task_names]

        # Define spaces: 39 (Base) + 3 (One-Hot)
        dummy_env = self.benchmark.train_classes[self.task_names[0]]()
        obs_shape = dummy_env.observation_space.shape[0] + len(self.task_names)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        self.action_space = dummy_env.action_space
        self.active_env = None
        self.current_task_idx = 0

    def _get_one_hot(self, task_idx):
        one_hot = np.zeros(len(self.task_names), dtype=np.float32)
        one_hot[task_idx] = 1.0
        return one_hot

    def reset(self, seed=None, options=None):
        task = random.choice(self.all_tasks)
        self.current_task_idx = self.task_names.index(task.env_name)

        env_cls = self.benchmark.train_classes[task.env_name]
        if self.active_env is None or self.active_env.__class__ != env_cls:
            self.active_env = env_cls()
            
        self.active_env._freeze_rand_vec = False 
        self.active_env.set_task(task)

        raw_obs = self.active_env.reset(seed=seed)
        if isinstance(raw_obs, tuple): raw_obs = raw_obs[0]

        full_obs = np.concatenate([raw_obs, self._get_one_hot(self.current_task_idx)])
        return full_obs.astype(np.float32), {}

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.active_env.step(action)
        full_obs = np.concatenate([raw_obs, self._get_one_hot(self.current_task_idx)])
        
        task_name = self.task_names[self.current_task_idx]
        info["task_name"] = task_name
        is_success = info.get("success", 0) > 0
        
        if is_success:
            if "reach" in task_name:
                reward += 2.0
            elif "push" in task_name:
                reward += 200.0
            elif "pick-place" in task_name:
                reward += 5000.0 
    
        reward = np.log1p(reward)
        return full_obs.astype(np.float32), reward, terminated, truncated, info

    def render(self): return self.active_env.render()
    def close(self):
        if self.active_env is not None: self.active_env.close()

def make_env(rank, seed=42):
    def _init():
        env = MT3Wrapper()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
        return env
    set_random_seed(seed + rank)
    return _init
    
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func 

def cosine_annealing(initial_value, total_steps, warmup_steps=2_000_000):
    def func(progress_remaining):
        current_step = (1 - progress_remaining) * total_steps
        if current_step < warmup_steps:
            return initial_value
        
        # Cosine decay for the remaining steps
        remaining_steps = total_steps - warmup_steps
        progress = (current_step - warmup_steps) / remaining_steps
        return initial_value * 0.5 * (1 + np.cos(np.pi * progress))
    return func





if __name__ == "__main__":
    # --- SEED HANDLING ---
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    SEED = int(slurm_id) if slurm_id else 42
    RUN_NAME = f"MT3_seed{SEED}"

    # --- HYPERPARAMETERS ---
    NUM_ENV = 24
    TOTAL_TIMESTEPS = 4_100_000
    ALGORITHM = "SAC"
    EVAL_FREQ = 20000 // NUM_ENV
    N_EVAL_EPISODES = 15 
    CHECKPOINT_FREQ = 50000
    BASE_LOG_DIR = "./metaworld_logs/MT3"

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    os.makedirs("./metaworld_models", exist_ok=True)

    # --- ENV SETUP ---
    # Added VecNormalize for better stability across tasks
    env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENV)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    eval_env = SubprocVecEnv([make_env(99, SEED)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)

    success_callback = SuccessCallback(log_freq=10000)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=cosine_annealing(3e-4, TOTAL_TIMESTEPS ),         
        buffer_size=2_000_000,         
        learning_starts=40000,        
        batch_size=1024,               
        tau=0.0005,
        gamma=0.99,                    
        train_freq=1,
        gradient_steps=2,        
        #sde_sample_freq=64,            
        ent_coef='auto_0.1',          
        target_entropy=-4.0,
        policy_kwargs=dict(
            net_arch=[512, 512, 512],  
            activation_fn=torch.nn.ReLU,
            optimizer_kwargs=dict(eps=5e-5)
        ),
        tensorboard_log=BASE_LOG_DIR,
        verbose=1,
        device=TARGET_DEVICE,
        seed=SEED,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_MT3/{RUN_NAME}/",
        name_prefix=f"sac_mt3",
        verbose=1
    )

    eval_callback = EvalSuccessCallback(
        eval_env,
        best_model_save_path= f"./metaworld_models/{RUN_NAME}/best_model/",
        log_path= f"{BASE_LOG_DIR}/{RUN_NAME}/eval/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, success_callback],
        log_interval=10,
        progress_bar=True,
        tb_log_name=RUN_NAME
    )

    print("\nSaving final model...")
    model.save(f"./metaworld_models/{RUN_NAME}_final")
    # Save the normalization stats as well
    env.save(f"./metaworld_models/{RUN_NAME}_vecnormalize.pkl")

    env.close()


    
########## For cluster  


 # run this locally 

#  ssh -L 6006:slurm-head-4:6006 e12434694@cluster.datalab.tuwien.ac.at


# for slurm, run this on slurm  Need to activate environment first. # micromamba activate robotlearning
 # python3 -m tensorboard.main --logdir="/home/e12434694/Robotlearning/Project/metaworld_logs/MT3" --port 6006 --bind_all

 # then this http://localhost:6006 
