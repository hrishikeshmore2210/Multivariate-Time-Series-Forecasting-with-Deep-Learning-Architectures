import os, time, argparse, yaml
import numpy as np
import torch
import torch.nn as nn

from src.utils import set_seed, get_device
from src.model_registry import create_model


def measure_latency(model, sample, device, warmup=20, iters=100):
    model.eval()
    sample = sample.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
    torch.cuda.synchronize() if device.type == "cuda" else None
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(sample)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return np.mean(times), np.std(times)

def main(cfg_path, ckpt_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    features = cfg["data"]["features"]
    seq_len = cfg["data"].get("seq_len", 50)
    batch_size = cfg["train"].get("batch_size", 64)
    output_dim = cfg["model"].get("output_dim", 1)

    # Dummy batch for latency based on feature size/seq_len
    dummy = torch.randn(batch_size, seq_len, len(features))

    model_kwargs = cfg["model"].copy()
    name = model_kwargs.pop("name")
    model = create_model(name, input_dim=len(features), **model_kwargs).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    mean_t, std_t = measure_latency(model, dummy, device)
    print(f"Latency per batch (mean±std): {mean_t*1000:.3f}±{std_t*1000:.3f} ms on {device}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="")
    a = p.parse_args()
    main(a.config, a.ckpt)
