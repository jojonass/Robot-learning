import os
os.environ["MUJOCO_GL"] = "egl"
import torch
import metaworld
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from utlitlies import *
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


# ==================== SETTINGS ==================== 
TASK_NAME = 'reach-v3'   # 'reach-v3', 'push-v3', 'pick-place-v3'
SEED = 1 # 1,2,3,4,5,6,7,8,8,10 seeds to check randomization of start 
NUM_ENV = 8
TOTAL_TIMESTEPS = 500_000

def make_env(rank, seed=42):
    def _init():
        mt1 = metaworld.MT1(TASK_NAME)
        env = mt1.train_classes[TASK_NAME]()
        specific_task = mt1.train_tasks[0]
        task_list = mt1.train_tasks
        specific_task = task_list[rank % len(task_list)]
        env.set_task(specific_task)
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    
        class SuccessFixer(gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                info["task_name"] = TASK_NAME 
                return obs, reward, terminated, truncated, info


        set_random_seed(seed + rank)
        return SuccessFixer(env)
        
    return _init

if __name__ == "__main__":
    

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id:
        SEED = int(slurm_task_id)
        print(f"Running on Cluster: Slurm Array Task ID {SEED} detected.")
    else:
        SEED = 1
        print("Running Locally: Defaulting to SEED 1.")
                
    if torch.cuda.is_available():
        TARGET_DEVICE = "cuda"
        print(f" CUDA available! Training will run on: {torch.cuda.get_device_name(0)}")
    else:
        TARGET_DEVICE = "cpu"
        print(" CUDA not available. Training will default to CPU.")
    print("=" * 60)
    
    # 1. Setup Environments
    env = SubprocVecEnv([make_env(i, SEED) for i in range(NUM_ENV)])
    eval_env = SubprocVecEnv([make_env(99, SEED)])

    BASE_LOG_DIR = "./metaworld_logs/MT1"
    RUN_NAME = f"{TASK_NAME}_seed{SEED}"
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    os.makedirs("./metaworld_models", exist_ok=True)
    

   
    # 2. Initialize SAC with TARGET_DEVICE
    model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,           # Reduced from 3e-4 to prevent policy collapse
    buffer_size=1_000_000,
    learning_starts=10000,        # Early start is fine for simple Reach
    batch_size=256,               # Higher batch size for smoother gradient updates
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=2,             # Set to 1 for MT1 to avoid over-optimizing on noise
    ent_coef='auto_1.0',          # Standard starting entropy
    target_entropy=-4.0,          # The standard for 4-dim action space
    policy_kwargs=dict(
        net_arch=[256, 256],      # Smaller net is actually MORE stable for Reach MT1
        activation_fn=torch.nn.ReLU 
    ),
    tensorboard_log=BASE_LOG_DIR,
    verbose=1,
    seed=SEED,
    device=TARGET_DEVICE 
)
    

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"./metaworld_models/checkpoints_{TASK_NAME}/",
        name_prefix=f"sac_{TASK_NAME}"
    )

    eval_callback = EvalSuccessCallback(
        eval_env,
        best_model_save_path= f"./metaworld_models/{RUN_NAME}/best_model/",
        log_path= f"{BASE_LOG_DIR}/{RUN_NAME}/eval/",
        eval_freq=10000 // NUM_ENV,
        n_eval_episodes=50,
        deterministic=True
    )

    # 4. Start Learning
    print(f"Starting MT1 training for task: {TASK_NAME}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=RUN_NAME
    )

    model.save(f"./metaworld_models/{RUN_NAME}_final")
    env.close()
    print("Training Complete.")


        # Locally run not on slurm

    # to check tensorboard: tensorboard --logdir="C:\Users\josef\OneDrive\Desktop\Robot learning folder\Project\metaworld_logs"


########## For cluster  


 # run this locally 

#  ssh -L 6006:slurm-head-4:6006 e12434694@cluster.datalab.tuwien.ac.at


# for slurm, run this on slurm  Need to activate environment first. # micromamba activate robotlearning
 # python3 -m tensorboard.main --logdir="/home/e12434694/Robotlearning/Project/metaworld_logs/MT1" --port 6006 --bind_all

 # then this http://localhost:6006 
