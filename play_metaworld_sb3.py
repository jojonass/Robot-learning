import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import metaworld
import numpy as np
from stable_baselines3 import SAC

# ==========================================
# 1. DYNAMIC OBSERVATION WRAPPER
# ==========================================
class DynamicMTWrapper(gym.ObservationWrapper):
    def __init__(self, env, task_index, total_tasks):
        super().__init__(env)
        self.task_index = task_index
        self.total_tasks = total_tasks 
        obs_shape = 39 + self.total_tasks
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

    def observation(self, obs):
        if self.total_tasks == 0: return obs.astype(np.float32)
        one_hot = np.zeros(self.total_tasks, dtype=np.float32)
        one_hot[self.task_index] = 1.0
        return np.concatenate([obs, one_hot]).astype(np.float32)

    def set_task(self, task):
        return self.env.set_task(task)

    @property
    def _freeze_rand_vec(self):
        return self.env._freeze_rand_vec

    @_freeze_rand_vec.setter
    def _freeze_rand_vec(self, value):
        self.env._freeze_rand_vec = value

# ==========================================
# 2. CAMERA FIX WRAPPER (Universal Version)
# ==========================================
class MetaWorldCameraWrapper(gym.Wrapper):
    def __init__(self, env, camera_name="topview"):
        super().__init__(env)
        self.camera_name = camera_name
        
        # Mapping common names to MuJoCo IDs if the name lookup fails
        self.cam_map = {
            'topview': 0,
            'corner': 1,
            'corner2': 2,
            'corner3': 3,
            'behindview': 4
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Set the name at the unwrapped level
        self.env.unwrapped.camera_name = self.camera_name
        return obs, info

    def render(self):
        # We reach into the low-level MuJoCo renderer
        # This forces the 'offscreen' camera used for recording
        if hasattr(self.env.unwrapped, 'mujoco_renderer'):
            renderer = self.env.unwrapped.mujoco_renderer
            # Force the camera ID into the renderer's config
            cam_id = self.cam_map.get(self.camera_name, 0)
            renderer.camera_id = cam_id
            renderer.camera_name = self.camera_name
            
        return self.env.render()

    def set_task(self, task):
        return self.env.set_task(task)

    @property
    def _freeze_rand_vec(self):
        return self.env._freeze_rand_vec

    @_freeze_rand_vec.setter
    def _freeze_rand_vec(self, value):
        self.env._freeze_rand_vec = value

# ==========================================
# 3. MAIN EVALUATION FUNCTION
# ==========================================
def run_evaluation():
    MODE = "MT10" 
    CHOSEN_CAMERA = "topview" # Options: 'topview', 'corner2', 'behindview'
    
    MT3_TASKS = ['reach-v3', 'push-v3', 'pick-place-v3']
    BASE_PATH = "/home/e12434694/Robotlearning/Project/metaworld_models"
    VIDEO_DIR = "./eval_videos"
    SEED = 42

    benchmark = metaworld.MT10(seed=SEED)
    num_one_hot = 10
    model_folder = "best_MT10"

    model_path = os.path.join(BASE_PATH, model_folder, "best_model.zip")
    print(f"🚀 Loading {MODE} model. Target Camera: {CHOSEN_CAMERA}")

    for i, (name, env_cls) in enumerate(benchmark.train_classes.items()):
        # Initialize
        raw_env = env_cls(render_mode='rgb_array')
        
        # Stack Wrappers
        mt_env = DynamicMTWrapper(raw_env, task_index=i, total_tasks=10)
        cam_env = MetaWorldCameraWrapper(mt_env, camera_name="corner")   
        
        # Set Task
        task_config = [t for t in benchmark.train_tasks if t.env_name == name][0]
        cam_env.set_task(task_config)

        # Record Video
        env = RecordVideo(
            cam_env, 
            video_folder=os.path.join(VIDEO_DIR, name),
            episode_trigger=lambda x: x == 0,
            name_prefix=f"eval_fixed"
        )

        if i == 0:
            model = SAC.load(model_path, env=env)

        # Execution
        obs, _ = env.reset()
        for _ in range(500): # Run one full episode (max_path_length is usually 500)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        print(f"Finished recording: {name}")
        env.close()
        
if __name__ == "__main__":
    run_evaluation()
