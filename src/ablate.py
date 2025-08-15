import os, argparse, yaml, copy, subprocess, sys


def run(cfg_path):
    subprocess.run([sys.executable, "src/train.py", "--config", cfg_path], check=True)

def main(cfg_path, variant):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_run = cfg["run_name"]

    if variant == "drop_transformer":
        cfg2 = copy.deepcopy(cfg)
        cfg2["run_name"] = base_run + "_ablate_no_trans"
        cfg2["model"]["name"] = "hybrid"
        cfg2["model"]["num_transformer_layers"] = 0
        tmp_path = "configs/_tmp_ablate_no_trans.yaml"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg2, f)
        run(tmp_path)

    elif variant == "drop_lstm":
        cfg2 = copy.deepcopy(cfg)
        cfg2["run_name"] = base_run + "_ablate_no_lstm"
        cfg2["model"]["name"] = "transformer"
        tmp_path = "configs/_tmp_ablate_no_lstm.yaml"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg2, f)
        run(tmp_path)

    else:
        raise ValueError("Unknown variant. Use: drop_transformer | drop_lstm")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True, choices=["drop_transformer", "drop_lstm"])
    a = p.parse_args()
    main(a.config, a.variant)
