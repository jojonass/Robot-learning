import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import metaworld
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==================== CONFIGURATION ====================
BASE_LOG_DIR = "/home/e12434694/Robotlearning/Project/metaworld_logs/SAC_MT3_2"

# --- FOR MT3 (Manual Task Names) ---
benchmark = metaworld.MT50() 
TASK_NAMES = ['reach-v3', 'push-v3', 'pick-place-v3']

# --- FOR MT10 (Uncomment for MT10 runs) and comment MT3
#benchmark = metaworld.MT10()
#TASK_NAMES = list(benchmark.train_classes.keys()) 

NUM_TEST_GOALS = 50
RAW_RESULTS_CSV = os.path.join(BASE_LOG_DIR, "detailed_task_results.csv")
SUMMARY_CSV = os.path.join(BASE_LOG_DIR, "final_statistics_summary.csv")

# ==================== SELECTIVE NORMALIZER ====================
class SelectiveVecNormalize(VecNormalize):
    def __init__(self, venv, norm_dim=39, **kwargs):
        super().__init__(venv, **kwargs)
        self.norm_dim = norm_dim

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.norm_obs: return obs
        obs_to_norm = obs[:, :self.norm_dim]
        extra_part = obs[:, self.norm_dim:]
        mean, var = self.obs_rms.mean[:self.norm_dim], self.obs_rms.var[:self.norm_dim]
        normalized = np.clip((obs_to_norm - mean) / np.sqrt(var + self.epsilon), -self.clip_obs, self.clip_obs)
        return np.concatenate([normalized, extra_part], axis=1).astype(np.float32)

def get_normalized_obs(raw_obs, task_idx, scaler, num_tasks):
    obs_39 = raw_obs.reshape(1, 39) 
    dummy_task = np.zeros((1, num_tasks), dtype=np.float32)
    full_input = np.concatenate([obs_39, dummy_task], axis=1)
    normed_full = scaler.normalize_obs(full_input)[0]
    one_hot = np.zeros(num_tasks, dtype=np.float32)
    one_hot[task_idx] = 1.0
    return np.concatenate([normed_full[:39], one_hot]).astype(np.float32)

# ==================== EVALUATION LOOP ====================
if __name__ == "__main__":
    num_tasks = len(TASK_NAMES)
    total_dims = 39 + num_tasks
    
    class MockEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_dims,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    
    dummy_vec = DummyVecEnv([lambda: MockEnv()])
    seed_dirs = sorted(glob.glob(os.path.join(BASE_LOG_DIR, "SAC_MT*_seed*")))
    
    rows = [] # To store every single trial result

    for s_path in seed_dirs:
        seed_id = os.path.basename(s_path)
        print(f"\n>>> EVALUATING: {seed_id}")
        
        norm_path = os.path.join(s_path, "vecnormalize.pkl")
        model_path = os.path.join(s_path, f"{seed_id}_final.zip")
        
        if not os.path.exists(norm_path) or not os.path.exists(model_path):
            continue

        vn_scaler = SelectiveVecNormalize.load(norm_path, dummy_vec)
        vn_scaler.training = False 
        model = SAC.load(model_path)

        for t_idx, t_name in enumerate(TASK_NAMES):
            print(f"{t_name:<20} | ", end="", flush=True)
            env = benchmark.train_classes[t_name]()
            all_tasks = [t for t in benchmark.train_tasks if t.env_name == t_name]
            test_tasks = all_tasks[:NUM_TEST_GOALS]
            
            successes = 0
            for task in test_tasks:
                env.set_task(task)
                obs, _ = env.reset()
                task_won = False
                for _ in range(500):
                    eval_obs = get_normalized_obs(obs, t_idx, vn_scaler, num_tasks)
                    action, _ = model.predict(eval_obs, deterministic=True)
                    obs, _, _, _, info = env.step(action)
                    if info.get('success', 0) > 0:
                        task_won = True; break
                
                print("." if task_won else "x", end="", flush=True)
                if task_won: successes += 1
            
            sr = (successes / NUM_TEST_GOALS) * 100
            rows.append({"Seed": seed_id, "Task": t_name, "Success_Rate": sr})
            print(f" | {sr:>5.1f}%")

    # ==================== FINAL STATISTICS ====================
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RAW_RESULTS_CSV, index=False)

        # 1. TASK-CENTRIC: Average success per task (Across Seeds)
        # Includes the STD between the same task of different seeds
        task_stats = df.groupby("Task")["Success_Rate"].agg(['mean', 'std']).reset_index()
        task_stats.columns = ['Task', 'Mean_SR', 'Seed_to_Seed_STD']

        # 2. SEED-CENTRIC: Average success per seed (Across all Tasks)
        # Includes the STD between different tasks within that specific seed
        seed_stats = df.groupby("Seed")["Success_Rate"].agg(['mean', 'std']).reset_index()
        seed_stats.columns = ['Seed', 'Average_SR', 'Task_to_Task_STD']

        overall_mean = df["Success_Rate"].mean()

        print("\n" + "="*75)
        print(f"{'TASK NAME':<25} | {'MEAN SR':>10} | {'STD (ACROSS SEEDS)':>20}")
        print("-" * 75)
        for _, row in task_stats.iterrows():
            std_val = row['Seed_to_Seed_STD'] if not pd.isna(row['Seed_to_Seed_STD']) else 0.0
            print(f"{row['Task']:<25} | {row['Mean_SR']:>9.2f}% | {std_val:>18.2f}%")
        
        print("\n" + "="*75)
        print(f"{'SEED ID':<25} | {'AVG SR':>10} | {'STD (BETWEEN TASKS)':>20}")
        print("-" * 75)
        for _, row in seed_stats.iterrows():
            std_val = row['Task_to_Task_STD'] if not pd.isna(row['Task_to_Task_STD']) else 0.0
            print(f"{row['Seed']:<25} | {row['Average_SR']:>9.2f}% | {std_val:>18.2f}%")

        print("-" * 75)
        print(f"OVERALL ENSEMBLE MEAN: {overall_mean:.2f}%")
        print("="*75)

        # Save both to the Summary CSV
        with open(SUMMARY_CSV, 'w') as f:
            f.write("--- TASK PERFORMANCE (Seed-to-Seed Stability) ---\n")
            task_stats.to_csv(f, index=False)
            f.write("\n--- SEED PERFORMANCE (Task-to-Task Consistency) ---\n")
            seed_stats.to_csv(f, index=False)
            f.write(f"\nTOTAL ENSEMBLE MEAN,{overall_mean}\n")
        
        print(f"\nAll averages and standard deviations saved to {SUMMARY_CSV}")

    # ==================== VISUALIZATION ====================
    try:
        import matplotlib.pyplot as plt

        # 1. Plot: Average Success Rate per TASK (Across all Seeds)
        plt.figure(figsize=(12, 6))
        task_means = df.groupby("Task")["Success_Rate"].mean()
        bars = plt.bar(task_means.index, task_means.values, color='skyblue', edgecolor='black')
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

        plt.axhline(y=overall_mean, color='red', linestyle='--', label=f'Ensemble Mean: {overall_mean:.1f}%')
        plt.title(f"Success Rate per Task\n(Average across all seeds, N={NUM_TEST_GOALS} evals/task)")
        plt.ylabel("Success Rate (%)")
        plt.xlabel("Task Name")
        plt.ylim(0, 115) # Extra space for labels
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        task_plot_path = os.path.join(BASE_LOG_DIR, "task_comparison_plot.png")
        plt.savefig(task_plot_path)
        print(f"Saved task bar plot to: {task_plot_path}")

        # 2. Plot: Average Success Rate per SEED (Across all Tasks)
        plt.figure(figsize=(12, 6))
        seed_means = df.groupby("Seed")["Success_Rate"].mean()
        bars = plt.bar(seed_means.index, seed_means.values, color='lightgreen', edgecolor='black')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

        plt.axhline(y=overall_mean, color='red', linestyle='--', label=f'Ensemble Mean: {overall_mean:.1f}%')
        plt.title(f"Success Rate per Seed\n(Average across all tasks)")
        plt.ylabel("Success Rate (%)")
        plt.xlabel("Seed ID")
        plt.ylim(0, 115)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        seed_plot_path = os.path.join(BASE_LOG_DIR, "seed_comparison_plot.png")
        plt.savefig(seed_plot_path)
        print(f"Saved seed bar plot to: {seed_plot_path}")

    except Exception as e:
        print(f"Plotting failed: {e}")
