import os
import subprocess
import glob

CONFIG_DIR = "configs"
EXPERIMENTS_DIR = "experiments"

def run_cmd(cmd_list):
    print(f"\n>>> Running: {' '.join(cmd_list)}\n")
    subprocess.run(cmd_list, check=True)

def main():
    configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    
    for cfg in configs:
        cfg_name = os.path.splitext(os.path.basename(cfg))[0]
        exp_dir = os.path.join(EXPERIMENTS_DIR, cfg_name)
        ckpt_path = os.path.join(exp_dir, "best.ckpt")
        pred_csv = os.path.join(exp_dir, "test_predictions.csv")
        err_dir = os.path.join(exp_dir, "error_analysis")

        if not os.path.exists(ckpt_path):
            print(f"⚠️  Skipping {cfg_name} — no best.ckpt found")
            continue

        # 1️⃣ Evaluation (metrics)
        run_cmd(["python", "-m", "src.eval", "--config", cfg, "--ckpt", ckpt_path])

        # 2️⃣ Latency
        run_cmd(["python", "-m", "src.latency", "--config", cfg, "--ckpt", ckpt_path])

        # 3️⃣ Error Analysis
        os.makedirs(err_dir, exist_ok=True)
        if os.path.exists(pred_csv):
            run_cmd(["python", "-m", "src.error_analysis", "--pred_csv", pred_csv, "--out_dir", err_dir])
        else:
            print(f"⚠️  Skipping error analysis for {cfg_name} — no test_predictions.csv found")

if __name__ == "__main__":
    main()
