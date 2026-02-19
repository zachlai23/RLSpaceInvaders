import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the completed 10,000 iteration dataset
try:
    script_dir = Path(__file__).resolve().parent
    candidate_csvs = [
        script_dir / "ppo_progress_real.csv",
        Path("ppo_progress_real.csv"),
    ]

    csv_path = next((p for p in candidate_csvs if p.exists()), None)
    if csv_path is None:
        checked = ", ".join(str(p) for p in candidate_csvs)
        raise FileNotFoundError(f"No progress CSV found. Checked: {checked}")

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plot the Raw Reward (Light blue with transparency)
    # Using df['iteration'] as X-axis ensures the scale goes to 10,000
    plt.plot(df['iteration'], df['reward'], color='skyblue', alpha=0.3, label='Raw Reward (per log)')
    
    # 2. Plot the Smoothed Learning Trend (Thick red line)
    # Applying a rolling window of 20 to highlight the overall progress
    plt.plot(df['iteration'], df['reward'].rolling(window=20).mean(), color='red', linewidth=2, label='PPO Learning Trend')
    
    # 3. Add the Random Baseline (Measured at 3.1)
    plt.axhline(y=3.1, color='green', linestyle='--', label='Random Baseline (3.1)')
    
    # Professional Chart Formatting
    plt.title("Advanced PPO Performance: 10,000 Iteration Analysis", fontsize=14)
    plt.xlabel("Training Iterations (Total Horizon: 10,000)", fontsize=12)
    plt.ylabel("Average Reward (128-step samples)", fontsize=12)
    
    # Set X-axis limits to clearly show the 10k span
    plt.xlim(0, 10000)
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Save the professional, English-only figure for the report
    output_path = script_dir / "final_10k_learning_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"Success! The professional chart has been saved as '{output_path.name}'")
    plt.show()

except Exception as e:
    print(f"Error during plotting: {e}")
