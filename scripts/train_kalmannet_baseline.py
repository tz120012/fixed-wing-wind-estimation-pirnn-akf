"""KalmanNet-style neural-augmented Kalman filter baseline for 3-D wind estimation.

This baseline addresses the reviewer request for a learning-based / neural-augmented
Kalman filter comparator (KalmanNet family, Revach et al., IEEE T-SP 2022).

Setup (model-based + learned gain, canonical KalmanNet):
  * State: 3-D wind vector w = [w_N, w_E, w_D] (m/s).
  * State-evolution model: random walk  x_k = x_{k-1}  (F = I).
  * Observation: kinematic wind pseudo-measurement
        z_k = v_g - TAS * [cos(theta)cos(psi), cos(theta)sin(psi), -sin(theta)]
    (identical convention to the AKF / EKF front-end of this project; H = I).
  * Kalman gain K_k is produced by a recurrent network from the standard KalmanNet
    feature set (innovation, observation difference, state-update difference) instead
    of an analytic covariance recursion.

The estimator therefore uses exactly the same physical observables as PX4-EKF2
(ground velocity, attitude, airspeed) and is trained end-to-end against the same
NED wind labels used by all other models, on the same data splits and seeds.
Predictions are written in m/s NED, window order identical to X_{split}.npy, so they
drop straight into the existing Table 1 / Table 4 evaluation pipeline.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# feature columns needed to build the kinematic measurement
KIN_IDX = [0, 1, 2, 9, 10, 11, 19]  # vel_n, vel_e, vel_d, roll, pitch, yaw, airspeed


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kinematic_measurement(X: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Build per-step kinematic wind measurement z (B, T, 3) from normalized X (B, T, 45)."""
    phys = X[..., KIN_IDX] * scale + mean       # (B, T, 7) -> vg(3), roll, pitch, yaw, tas
    phys = torch.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)
    vg = phys[..., 0:3]
    pitch = phys[..., 4]
    yaw = phys[..., 5]
    tas = phys[..., 6]
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    v_air = torch.stack([tas * cp * cy, tas * cp * sy, -tas * sp], dim=-1)
    return vg - v_air


class KalmanNet(nn.Module):
    """Random-walk state model with a GRU-learned Kalman gain (KalmanNet architecture)."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(9, hidden_size)          # features: innov(3), dz(3), dx(3)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 9),                 # 3x3 Kalman gain
        )
        # zero-init the gain output: initial gain ~ 0 -> stable recursion (x stays at z_0)
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, 3) kinematic measurements -> final wind estimate (B, 3)."""
        b, t, _ = z.shape
        h = torch.zeros(b, self.hidden_size, device=z.device, dtype=z.dtype)
        x_prev = z[:, 0, :]
        x_prev2 = x_prev
        z_prev = z[:, 0, :]
        for k in range(1, t):
            z_k = z[:, k, :]
            innov = z_k - x_prev                       # F2 innovation
            dz = z_k - z_prev                          # F4 observation difference
            dx = x_prev - x_prev2                      # F1 state-update difference
            feat = torch.cat([innov, dz, dx], dim=-1)
            h = self.gru(feat, h)
            kg = 2.0 * torch.tanh(self.gain_head(h)).view(b, 3, 3)   # bounded gain for stability
            x_new = x_prev + torch.bmm(kg, innov.unsqueeze(-1)).squeeze(-1)
            x_new = torch.clamp(x_new, -60.0, 60.0)
            x_prev2 = x_prev
            x_prev = x_new
            z_prev = z_k
        return x_prev


def iterate_batches(n: int, bs: int, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def predict(model, X, mean, scale, device, bs=8192) -> np.ndarray:
    model.eval()
    out = np.empty((len(X), 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device).float()
            z = kinematic_measurement(xb, mean, scale)
            out[i:i + bs] = model(z).cpu().numpy()
    return out


def train_seed(seed, X_train, y_tgt_train, X_val, y_tgt_val, mean, scale,
               device, epochs, batch_size, lr, patience):
    set_seed(seed)
    model = KalmanNet(hidden_size=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    yv = torch.from_numpy(y_tgt_val).to(device)
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        for bidx in iterate_batches(len(X_train), batch_size, True, rng):
            xb = torch.from_numpy(np.ascontiguousarray(X_train[bidx])).to(device).float()
            tb = torch.from_numpy(y_tgt_train[bidx]).to(device)
            z = kinematic_measurement(xb, mean, scale)
            pred = model(z)
            loss = loss_fn(pred, tb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        # validation RMSE
        model.eval()
        with torch.no_grad():
            vpred = []
            for i in range(0, len(X_val), 8192):
                xb = torch.from_numpy(np.ascontiguousarray(X_val[i:i + 8192])).to(device).float()
                vpred.append(model(kinematic_measurement(xb, mean, scale)))
            vpred = torch.cat(vpred, 0)
            val_rmse = torch.sqrt(loss_fn(vpred, yv)).item()
        sched.step(val_rmse)
        print(f"  seed {seed} ep {ep + 1:02d}/{epochs}  train_mse={tot / max(nb,1):.4f}  "
              f"val_rmse={val_rmse:.4f} m/s  ({time.time() - t0:.1f}s)")
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  seed {seed}: early stop at epoch {ep + 1} (best val_rmse={best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/dataset_new_processed")
    ap.add_argument("--out-dir", default="data/baseline_kalmannet")
    ap.add_argument("--seeds", default="26,42,2026")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--splits", default="test_id,test_ood")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / args.data_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pickle.load(open(data_dir / "norm_params.pkl", "rb"))
    sx, sy = meta["scaler_X"], meta["scaler_y"]
    mean = torch.tensor(sx.mean_[KIN_IDX], device=device, dtype=torch.float32)
    scale = torch.tensor(sx.scale_[KIN_IDX], device=device, dtype=torch.float32)
    y_mean = sy.mean_[:3].astype(np.float32)
    y_scale = sy.scale_[:3].astype(np.float32)

    print("loading arrays ...")
    X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(data_dir / "y_val.npy")
    # targets in physical m/s NED
    y_tgt_train = (y_train[:, 0:3] * y_scale + y_mean).astype(np.float32)
    y_tgt_val = (y_val[:, 0:3] * y_scale + y_mean).astype(np.float32)
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}")

    splits = args.splits.split(",")
    split_X = {s: np.load(data_dir / f"X_{s}.npy", mmap_mode="r") for s in splits}

    seeds = [int(s) for s in args.seeds.split(",")]
    summary = []
    for seed in seeds:
        print(f"\n=== training KalmanNet seed {seed} ===")
        model, best_val = train_seed(
            seed, X_train, y_tgt_train, X_val, y_tgt_val, mean, scale,
            device, args.epochs, args.batch_size, args.lr, args.patience,
        )
        torch.save({"model_state_dict": model.state_dict(), "seed": seed, "best_val_rmse": best_val},
                   out_dir / f"kalmannet_seed{seed}.pth")
        for s in splits:
            pred = predict(model, split_X[s], mean, scale, device)
            np.save(out_dir / f"seed{seed}_{s}_kalmannet.npy", pred)
            print(f"  saved predictions: seed{seed}_{s}_kalmannet.npy  {pred.shape}")
        summary.append((seed, best_val))

    print("\n=== KalmanNet training summary (val RMSE m/s) ===")
    for seed, v in summary:
        print(f"  seed {seed}: {v:.4f}")
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
