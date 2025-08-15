import os, argparse, numpy as np
import matplotlib.pyplot as plt


def main(pred_csv, out_dir, worst_k=50):
    os.makedirs(out_dir, exist_ok=True)
    arr = np.loadtxt(pred_csv, delimiter=",")
    # assumes columns: [gt, pred] (or [gt_dim..., pred_dim...]); here we take first column as gt, last as pred
    gt = arr[:, 0]
    pred = arr[:, -1]
    err = pred - gt
    abs_err = np.abs(err)

    # Save sorted worst cases
    idxs = np.argsort(-abs_err)[:worst_k]
    worst = np.stack([gt[idxs], pred[idxs], err[idxs], abs_err[idxs]], axis=1)
    np.savetxt(os.path.join(out_dir, "worst_cases.csv"), worst, delimiter=",", fmt="%.6f")

    # Histograms
    plt.figure()
    plt.hist(err, bins=50)
    plt.title("Residuals (pred - gt)")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir, "residuals_hist.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.hist(abs_err, bins=50)
    plt.title("Absolute Error")
    plt.xlabel("|Error|")
    plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir, "abs_error_hist.png"), dpi=150)
    plt.close()

    print(f"Saved error analysis to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--worst_k", type=int, default=50)
    a = p.parse_args()
    main(a.pred_csv, a.out_dir, a.worst_k)
