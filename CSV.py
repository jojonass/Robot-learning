import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from collections import defaultdict

def export_combined_task_stats(task_path, output_base="CSV_Data"):
    task_name = os.path.basename(task_path.rstrip('/'))
    task_output_folder = os.path.join(output_base, task_name)
    
    os.makedirs(task_output_folder, exist_ok=True)
    print(f"Exporting all metrics for task: {task_name}")
    
    data_by_metric = defaultdict(list)
    
    seed_folders = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
    
    for seed_name in seed_folders:
        seed_path = os.path.join(task_path, seed_name)
        print(f"  -> Reading Seed: {seed_name}")
        
        event_acc = EventAccumulator(seed_path)
        event_acc.Reload()
        
        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            
            df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', seed_name])
            
            # --- FIX STARTS HERE ---
            # Remove duplicate steps (keep the last one recorded)
            df = df.drop_duplicates(subset=['step'], keep='last')
            # -----------------------
            
            df.set_index('step', inplace=True)
            
            clean_tag = tag.replace('/', '_')
            data_by_metric[clean_tag].append(df)

    print("\nMerging seeds into single files...")
    for metric, df_list in data_by_metric.items():
        try:
            # Join all seed columns into one table
            combined_df = pd.concat(df_list, axis=1)
            
            # Add Mean column
            combined_df['mean'] = combined_df.mean(axis=1)
            
            save_path = os.path.join(task_output_folder, f"{metric}.csv")
            combined_df.to_csv(save_path)
            print(f"  Successfully saved: {metric}.csv")
        except Exception as e:
            print(f"  Error merging {metric}: {e}")

# --- EXECUTION ---
TASK_DIR = "/home/e12434694/Robotlearning/Project/metaworld_logs/SAC_MT3_2"
export_combined_task_stats(TASK_DIR)
