import os, argparse, json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device, save_json
from src.data import TimeSeriesWindowDataset, compute_standard_scaler
from src.model_registry import create_model
from src.metrics import mae, rmse, mape, smape


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, gts = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        preds.append(pred.detach().cpu().numpy())
        gts.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return (total_loss / len(loader.dataset)), preds, gts

def main(cfg_path, ckpt_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    te_csv = cfg["data"]["test_csv"]
    features = cfg["data"]["features"]
    target = cfg["data"]["target"]
    seq_len = cfg["data"].get("seq_len", 50)
    stride = cfg["data"].get("stride", 1)
    output_dim = cfg["model"].get("output_dim", 1)

    scaler = compute_standard_scaler(cfg["data"]["train_csv"], features)
    test_ds = TimeSeriesWindowDataset(te_csv, features, target, seq_len, stride, scaler, output_dim)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"].get("batch_size", 64), shuffle=False, num_workers=2)

    model_kwargs = cfg["model"].copy()
    name = model_kwargs.pop("name")
    model = create_model(name, input_dim=len(features), **model_kwargs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.MSELoss()
    te_loss, te_pred, te_gt = evaluate(model, test_loader, criterion, device)

    out_dir = os.path.join("experiments", cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "test_mse": float(te_loss),
        "test_mae": float(mae(te_gt, te_pred)),
        "test_rmse": float(rmse(te_gt, te_pred)),
        "test_mape": float(mape(te_gt, te_pred)),
        "test_smape": float(smape(te_gt, te_pred)),
    }
    save_json(metrics, os.path.join(out_dir, "metrics_eval.json"))
    np.savetxt(os.path.join(out_dir, "test_predictions.csv"), np.hstack([te_gt, te_pred]), delimiter=",", fmt="%.6f")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    a = p.parse_args()
    main(a.config, a.ckpt)
