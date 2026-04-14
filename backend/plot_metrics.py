import pandas as pd
import matplotlib.pyplot as plt

def generate_analytics_report():
    print("📊 ClearDrive Analytics Engine Booting...")
    filename = "performance_results.csv"

    try:
        # CSV file ko read karna
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ Error: '{filename}' nahi mili. Pehle main.py ko run karo taaki data generate ho!")
        return

    # Column names se extra spaces hatana (just to be safe)
    df.columns = df.columns.str.strip()
    
    # Check karna ki data hai ya nahi
    if len(df) == 0:
        print("❌ Error: CSV file khali hai. main.py ko thodi der chalne do.")
        return

    # Graph ka overall style aur size set karna
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('ClearDrive System Performance Analytics', fontsize=18, fontweight='bold', color='#2c3e50')

    # X-axis (Frame/Time progression)
    frames = range(len(df))

    # --- Graph 1: System Stability (FPS over time) ---
    axes[0].plot(frames, df['FPS'], color='#2980b9', linewidth=2.5, label='Actual FPS')
    # Ek red line jo 20 FPS (Target) dikhayegi
    axes[0].axhline(y=20, color='#c0392b', linestyle='--', alpha=0.8, label='Target Real-time (20 FPS)')
    axes[0].set_title('System Latency & Hardware Stability (RTX 3050)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frames Per Second (FPS)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.4)

    # --- Graph 2: Image Restoration (Contrast Gain) ---
    axes[1].plot(frames, df['Contrast_Gain'], color='#27ae60', linewidth=2.5, label='Contrast Improvement (%)')
    axes[1].set_title('DIP Engine: Atmospheric Restoration Efficiency', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Contrast Gain (%)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.4)

    # --- Graph 3: Safety Validation (Visibility Score) ---
    axes[2].fill_between(frames, df['Visibility_Score'], color='#e67e22', alpha=0.3, label='Safe Zone Coverage')
    axes[2].plot(frames, df['Visibility_Score'], color='#d35400', linewidth=2.5)
    axes[2].set_title('ADAS Safety Metric: Live Visibility Mapping', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Safe Zone Area (%)')
    axes[2].set_xlabel('Processed Frames (Time)')
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.4)

    # Layout ko clean karna taaki text overlap na ho
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # PPT ke liye High-Quality Image save karna
    output_filename = 'cleardrive_metrics_report.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Graphs generate ho gaye hain aur '{output_filename}' save ho gayi hai.")

    # Screen par graph dikhana
    plt.show()

if __name__ == "__main__":
    generate_analytics_report()