import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import metaworld
from stable_baselines3 import SAC

# ==================== CONFIGURATION ====================
BASE_LOG_DIR = "/home/e12434694/Robotlearning/Project/metaworld_logs/SAC_pick-place-v3_1"
TASK_NAME = 'pick-place-v3'
NUM_TEST_GOALS = 100 # This is the number of evaluations per seed

DETAILED_CSV = os.path.join(BASE_LOG_DIR, "final_eval_detailed_seeds.csv")
SUMMARY_CSV = os.path.join(BASE_LOG_DIR, "final_eval_summary_stats.csv")

# ==================== EVALUATION LOOP ====================
if __name__ == "__main__":
    print(f"Initializing MetaWorld MT1 for {TASK_NAME}...")
    mt1 = metaworld.MT1(TASK_NAME)
    
    seed_dirs = sorted(glob.glob(os.path.join(BASE_LOG_DIR, "*seed*")))

    if not seed_dirs:
        print(f"No seed directories found in {BASE_LOG_DIR}")
        exit()

    print(f"Found {len(seed_dirs)} seeds. Starting evaluation...")
    all_results = []

    for s_path in seed_dirs:
        seed_id = os.path.basename(s_path)
        model_path = os.path.join(s_path, f"{seed_id}_final.zip")
        
        if not os.path.exists(model_path):
            continue

        print(f"\n>>> SEED: {seed_id}")
        
        # Load to CPU to avoid resource conflict with your other running process
        model = SAC.load(model_path, device="cpu")
        
        env = mt1.train_classes[TASK_NAME]()
        available_tasks = mt1.train_tasks
        
        successes = 0
        print(f"Goals (N={NUM_TEST_GOALS}): ", end="", flush=True)

        for i in range(NUM_TEST_GOALS):
            task = available_tasks[i % len(available_tasks)]
            env.set_task(task)
            obs, _ = env.reset(seed=i) 
            
            task_won = False
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _, info = env.step(action)
                if info.get('success', 0) > 0:
                    task_won = True
                    break
            
            if task_won:
                successes += 1
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)
            
        env.close()
        sr = (successes / NUM_TEST_GOALS) * 100
        all_results.append({
            "Seed": seed_id, 
            "Success_Rate": sr, 
            "Evaluations": NUM_TEST_GOALS
        })
        print(f" | SR: {sr:.1f}%")

    # ==================== STATISTICS & EXPORT ====================
    if all_results:
        df = pd.DataFrame(all_results)
        mean_sr = df["Success_Rate"].mean()
        std_sr = df["Success_Rate"].std()
        
        print("\n" + "="*45)
        print(f"RESULTS FOR {TASK_NAME}")
        print(f"Mean SR: {mean_sr:.2f}% | Std: {std_sr:.2f}%")
        print(f"Episodes per seed: {NUM_TEST_GOALS}")
        print("="*45)
        
        # Save detailed CSV (includes 'Evaluations' column for each seed)
        df.to_csv(DETAILED_CSV, index=False)
        
        # Save summary CSV
        summary_df = pd.DataFrame([{
            "Task": TASK_NAME,
            "Mean_Success_Rate": mean_sr,
            "Std_Success_Rate": std_sr,
            "Seeds_Evaluated": len(df),
            "Evals_Per_Seed": NUM_TEST_GOALS
        }])
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved results to: {DETAILED_CSV}")

        # ==================== PLOTTING ====================
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            bars = plt.bar(df["Seed"], df["Success_Rate"], color='skyblue', edgecolor='black')
            
            # Add labels on top of bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

            plt.axhline(y=mean_sr, color='red', linestyle='--', label=f'Mean: {mean_sr:.1f}%')
            plt.title(f"Success Rate across Seeds - {TASK_NAME}\n(N={NUM_TEST_GOALS} evals per seed)")
            plt.ylabel("Success Rate (%)")
            plt.xlabel("Seed ID")
            plt.ylim(0, 110)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(BASE_LOG_DIR, "final_eval_plot.png")
            plt.savefig(plot_path)
            print(f"Saved bar plot to: {plot_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")
