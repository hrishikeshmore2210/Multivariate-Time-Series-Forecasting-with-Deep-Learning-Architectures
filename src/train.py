import os, argparse, json, time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, get_device, save_json, save_config_snapshot, count_parameters
from src.data import TimeSeriesWindowDataset, compute_standard_scaler
from src.model_registry import create_model
from src.metrics import mae, rmse, mape, smape


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, gts = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        preds.append(pred.detach().cpu().numpy())
        gts.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return (total_loss / len(loader.dataset)), preds, gts

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    run_name = cfg["run_name"]
    out_dir = os.path.join("experiments", run_name)
    os.makedirs(out_dir, exist_ok=True)
    save_config_snapshot(cfg, out_dir)

    # Data/scaler
    tr_csv = cfg["data"]["train_csv"]
    va_csv = cfg["data"]["val_csv"]
    te_csv = cfg["data"].get("test_csv", None)
    features = cfg["data"]["features"]
    target = cfg["data"]["target"]
    seq_len = cfg["data"].get("seq_len", 50)
    stride = cfg["data"].get("stride", 1)
    output_dim = cfg["model"].get("output_dim", 1)

    scaler = compute_standard_scaler(tr_csv, features)

    train_ds = TimeSeriesWindowDataset(tr_csv, features, target, seq_len, stride, scaler, output_dim)
    val_ds   = TimeSeriesWindowDataset(va_csv, features, target, seq_len, stride, scaler, output_dim)
    if te_csv:
        test_ds  = TimeSeriesWindowDataset(te_csv, features, target, seq_len, stride, scaler, output_dim)
    else:
        test_ds = None

    batch_size = cfg["train"].get("batch_size", 64)
    num_workers = cfg["train"].get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds else None

    # Model
    model_kwargs = cfg["model"].copy()
    name = model_kwargs.pop("name")
    model = create_model(name, input_dim=len(features), **model_kwargs).to(device)
    n_params = count_parameters(model)

    lr = cfg["train"].get("lr", 1e-3)
    weight_decay = cfg["train"].get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    epochs = cfg["train"].get("epochs", 20)
    patience = cfg["train"].get("patience", 5)
    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.ckpt")
    no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_pred, va_gt = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            torch.save({"model_state": model.state_dict()}, best_path)
            no_improve = 0
            # save current val predictions
            import numpy as np
            np.savetxt(os.path.join(out_dir, "val_predictions.csv"), np.hstack([va_gt, va_pred]), delimiter=",", fmt="%.6f")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping.")
            break

    # Final test evaluation on best checkpoint
    metrics = {"n_params": int(n_params), "best_val_loss": float(best_val)}
    if test_loader is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        te_loss, te_pred, te_gt = evaluate(model, test_loader, criterion, device)
        metrics["test_mse"] = float(te_loss)
        metrics["test_mae"] = float(mae(te_gt, te_pred))
        metrics["test_rmse"] = float(rmse(te_gt, te_pred))
        metrics["test_mape"] = float(mape(te_gt, te_pred))
        metrics["test_smape"] = float(smape(te_gt, te_pred))
        import numpy as np
        np.savetxt(os.path.join(out_dir, "test_predictions.csv"), np.hstack([te_gt, te_pred]), delimiter=",", fmt="%.6f")

    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    print("Done. Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
