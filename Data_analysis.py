import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tbparse import SummaryReader

# --- CONFIGURATION ---
LOG_DIR = r"C:\Users\josef\OneDrive\Desktop\Robot learning folder\Mt10_data\MT10"
# Save results in a new sub-folder
SAVE_DIR = r"C:\Users\josef\OneDrive\Desktop\Robot learning folder\Mt10_data\Results\Individual_Plots"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"🚀 Reading logs...")
reader = SummaryReader(LOG_DIR)
df = reader.scalars

if df.empty:
    print("❌ No data found.")
    exit()


def smooth_data(values, weight=0.85):
    if len(values) == 0: return values
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# --- LOOP THROUGH EVERY TAG ---
all_tags = df['tag'].unique()
print(f"📊 Found {len(all_tags)} unique metrics. Plotting them separately...")

for tag in all_tags:
    # 1. Prepare Data
    subset = df[df['tag'] == tag].copy()
    if len(subset) < 2:
        continue  # Skip tags with only 1 data point

    # 2. Smooth the line
    subset['value'] = smooth_data(subset['value'].values)

    # 3. Create the Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("white")  # Clean background for individual plots

    sns.lineplot(data=subset, x='step', y='value', color='royalblue', linewidth=2.5)

    # Clean up the title (removing slashes for the filename)
    clean_title = tag.replace('/', '_').replace('\\', '_')

    plt.title(f"Metric: {tag}", fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.tight_layout()

    # 4. Save
    file_path = os.path.join(SAVE_DIR, f"{clean_title}.png")
    plt.savefig(file_path, dpi=200)
    plt.close()  # CRITICAL: Closes the plot so memory doesn't leak
    print(f"✅ Saved: {clean_title}.png")

print(f"\n✨ ALL DONE! Check your folder:\n{SAVE_DIR}")