import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import metaworld
import numpy as np

from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ============================================================
# ======================== CONFIG ===========================
# ============================================================

# Map names to camera IDs
CAMERAS = {
    "default": 0,
    "upside down": 1,
    "corner": 2,
    "topview": 3,
}

# Pick camera
CAMERA = CAMERAS["default"]  # choose topview / corner / behindview

BENCHMARK = "MT10"             # MT1, MT3, MT10
TASKS = None         # None = all tasks, or subset like ["reach-v3"]
SEEDS = [1]                   # Seeds to evaluate
MAX_STEPS = 500
EXPERIMENT_NAME = "SAC_MT10_2"

MODEL_ROOT = "/home/e12434694/Robotlearning/Project/metaworld_logs"
VIDEO_ROOT = "./eval_videos"

# ============================================================
# ===================== CAMERA WRAPPER =======================
# ============================================================

class CameraWrapper(gym.Wrapper):
    def __init__(self, env, camera_index=0):
        super().__init__(env)
        self.camera_index = camera_index

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # set camera after reset
        if hasattr(self.env.unwrapped, "mujoco_renderer"):
            self.env.unwrapped.mujoco_renderer.camera_id = self.camera_index
        return obs, info

    def render(self, *args, **kwargs):
        # forward all kwargs to avoid TypeError
        return self.env.render(*args, **kwargs)

    def set_task(self, task):
        return self.env.set_task(task)

# ============================================================
# =================== ONE-HOT TASK WRAPPER ===================
# ============================================================

class OneHotTaskWrapper(gym.ObservationWrapper):
    def __init__(self, env, task_idx, num_tasks):
        super().__init__(env)
        self.task_idx = task_idx
        self.num_tasks = num_tasks
        obs_dim = env.observation_space.shape[0] + num_tasks
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros(self.num_tasks, dtype=np.float32)
        one_hot[self.task_idx] = 1.0
        return np.concatenate([obs, one_hot]).astype(np.float32)

    def set_task(self, task):
        return self.env.set_task(task)

# ============================================================
# ================== VECNORMALIZE LOADER =====================
# ============================================================

def load_vecnormalize(env, vecnorm_path):
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return env

# ============================================================
# ======================= MT1 RECORD ========================
# ============================================================

def record_mt1(seed, task_name):
    print(f"🎥 MT1 | {task_name} | seed {seed}")
    mt1 = metaworld.MT1(task_name, seed=seed)
    env_cls = mt1.train_classes[task_name]

    model_path = os.path.join(
        MODEL_ROOT, "SAC_MT1", task_name, f"SAC_MT1_seed{seed}", "best_model", "best_model.zip"
    )
    model = SAC.load(model_path)

    env = env_cls(render_mode="rgb_array")
    env = CameraWrapper(env, CAMERA)
    task = mt1.train_tasks[0]
    env.set_task(task)

    video_dir = os.path.join(VIDEO_ROOT, "MT1", task_name, f"seed{seed}")
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True, name_prefix=f"{task_name}_{CAMERA}")

    obs, _ = env.reset()
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    env.close()

# ============================================================
# ================= MT3 / MT10 RECORD =======================
# ============================================================

def record_mt_multitask(seed):
    print(f"\n🎥 {BENCHMARK} | seed {seed}")

    if BENCHMARK == "MT3":
        benchmark = metaworld.MT50(seed=seed)
        task_names = ['reach-v3', 'push-v3', 'pick-place-v3']
        tasks = [t for t in benchmark.train_tasks if t.env_name in task_names]
        env_classes = {k: benchmark.train_classes[k] for k in task_names}
    elif BENCHMARK == "MT10":
        benchmark = metaworld.MT10(seed=seed)
        tasks = benchmark.train_tasks
        env_classes = benchmark.train_classes
        task_names = list(env_classes.keys())
    else:
        raise ValueError("Invalid BENCHMARK")

    base_seed_dir = os.path.join(MODEL_ROOT, EXPERIMENT_NAME, f"SAC_{BENCHMARK}_seed{seed}")
    model_path = os.path.join(base_seed_dir, "best_model", "best_model.zip")
    vecnorm_path = os.path.join(base_seed_dir, "best_model", "vecnormalize.pkl")
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    assert os.path.exists(vecnorm_path), f"Missing VecNormalize: {vecnorm_path}"

    for task_idx, task_name in enumerate(task_names):
        if TASKS and task_name not in TASKS:
            continue
        print(f"  ▶ Task: {task_name}")

        env_cls = env_classes[task_name]
        raw_env = env_cls(render_mode="rgb_array")
        raw_env = OneHotTaskWrapper(raw_env, task_idx, len(task_names))
        raw_env = CameraWrapper(raw_env, CAMERA)

        task = [t for t in tasks if t.env_name == task_name][0]
        raw_env.set_task(task)

        video_dir = os.path.join(VIDEO_ROOT, BENCHMARK, f"seed{seed}", task_name)
        os.makedirs(video_dir, exist_ok=True)
        raw_env = RecordVideo(raw_env, video_folder=video_dir, episode_trigger=lambda e: True, name_prefix=f"{task_name}_{CAMERA}")

        env = DummyVecEnv([lambda: raw_env])
        env = load_vecnormalize(env, vecnorm_path)
        model = SAC.load(model_path, env=env)

        obs = env.reset()
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                break
        env.close()

# ============================================================
# =========================== MAIN ==========================
# ============================================================

if __name__ == "__main__":
    for seed in SEEDS:
        if BENCHMARK == "MT1":
            assert TASKS is not None, "MT1 requires explicit TASKS"
            for task in TASKS:
                record_mt1(seed, task)
        else:
            record_mt_multitask(seed)

    print("\n All videos recorded.")
