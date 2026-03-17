import json
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


@dataclass
class Settings:
    total_steps: int = 6000
    lookback: int = 60
    horizon: int = 10
    n_metrics: int = 4

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    batch_size: int = 128
    hidden_dim: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    early_stop: int = 5

    n_incidents: int = 18
    incident_min_len: int = 12
    incident_max_len: int = 35

    out_dir: str = "outputs"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_spikes(x: np.ndarray, prob: float, scale: float) -> np.ndarray:
    y = x.copy()
    mask = np.random.rand(len(y)) < prob
    if mask.sum() == 0:
        return y

    spikes = (np.random.pareto(3.0, size=mask.sum()) + 1.0) * scale
    signs = np.random.choice([-1.0, 1.0], size=mask.sum(), p=[0.2, 0.8])
    y[mask] += spikes * signs
    return y


def sample_incidents(total_steps: int, n_incidents: int, min_len: int, max_len: int, gap: int = 80):
    """
    Returns a sorted list of (start, end) intervals, inclusive.
    I keep a gap between incidents so the labels do not become too messy.
    """
    intervals = []
    tries = 0

    while len(intervals) < n_incidents and tries < 5000:
        tries += 1
        length = np.random.randint(min_len, max_len + 1)
        start = np.random.randint(120, total_steps - length - 20)
        end = start + length - 1

        overlaps = False
        for s, e in intervals:
            if not (end + gap < s or start - gap > e):
                overlaps = True
                break

        if not overlaps:
            intervals.append((start, end))

    intervals.sort()
    return intervals


def build_synthetic_series(cfg: Settings):
    """
    Synthetic cloud-ish metrics:
      - cpu
      - latency p95
      - error rate
      - request count

    I wanted something simple but still realistic enough to justify short-horizon alerting.
    """
    T = cfg.total_steps
    t = np.arange(T, dtype=np.float32)

    day = 24 * 60
    week = 7 * day

    load_pattern = (
        0.55
        + 0.18 * np.sin(2 * np.pi * t / day)
        + 0.06 * np.sin(2 * np.pi * t / (day / 2.0))
        + 0.04 * np.sin(2 * np.pi * t / week)
    )

    drift = 0.00003 * t

    req = 1000 + 300 * load_pattern + 40 * np.random.randn(T) + 80 * drift
    req = add_spikes(req, prob=0.003, scale=250)
    req = np.maximum(req, 50)

    cpu = 45 + 28 * load_pattern + 3 * np.random.randn(T) + 8 * drift
    cpu = add_spikes(cpu, prob=0.002, scale=10)
    cpu = np.clip(cpu, 0, 100)

    latency = 120 + 35 * load_pattern + 8 * np.random.randn(T)
    latency += 0.02 * np.maximum(req - 1100, 0)
    latency = add_spikes(latency, prob=0.003, scale=80)
    latency = np.maximum(latency, 1)

    error = 0.01 + 0.003 * np.random.randn(T) + 0.0004 * np.maximum(cpu - 70, 0)
    error += 0.0001 * np.maximum(latency - 180, 0)
    error = np.maximum(error, 0)

    incidents = sample_incidents(
        total_steps=T,
        n_incidents=cfg.n_incidents,
        min_len=cfg.incident_min_len,
        max_len=cfg.incident_max_len,
    )

    incident_mask = np.zeros(T, dtype=np.int64)

    for start, end in incidents:
        incident_mask[start:end + 1] = 1

        # Before the incident starts, I add a ramp so the task is not impossible.
        ramp_len = np.random.randint(8, 20)
        ramp_start = max(0, start - ramp_len)
        ramp = np.linspace(0.0, 1.0, start - ramp_start, endpoint=False) if start > ramp_start else np.array([])

        if len(ramp) > 0:
            cpu[ramp_start:start] += 10 * ramp + 1.5 * np.random.randn(len(ramp))
            latency[ramp_start:start] += 40 * ramp + 4 * np.random.randn(len(ramp))
            error[ramp_start:start] += 0.015 * ramp + 0.002 * np.random.randn(len(ramp))
            req[ramp_start:start] += 120 * ramp + 15 * np.random.randn(len(ramp))

        length = end - start + 1
        cpu[start:end + 1] += 12 + 6 * np.random.randn(length)
        latency[start:end + 1] += 100 + 25 * np.random.randn(length)
        error[start:end + 1] += 0.04 + 0.01 * np.random.randn(length)

        if np.random.rand() < 0.5:
            req[start:end + 1] += 200 + 40 * np.random.randn(length)
        else:
            req[start:end + 1] -= 180 + 30 * np.random.randn(length)

    cpu = np.clip(cpu, 0, 100)
    latency = np.maximum(latency, 1)
    error = np.clip(error, 0, 1)
    req = np.maximum(req, 1)

    X = np.stack([cpu, latency, error, req], axis=1).astype(np.float32)
    names = ["cpu_usage", "latency_p95", "error_rate", "request_count"]

    return X, incident_mask, incidents, names


def make_future_labels(incident_mask: np.ndarray, horizon: int) -> np.ndarray:
    """
    y[t] = 1 if an incident is active at least once in the next `horizon` steps.
    """
    y = np.zeros(len(incident_mask), dtype=np.int64)

    for t in range(len(incident_mask)):
        lo = t + 1
        hi = min(len(incident_mask), t + horizon + 1)
        if lo < hi and incident_mask[lo:hi].any():
            y[t] = 1

    return y


def make_windows(X: np.ndarray, y_future: np.ndarray, lookback: int):
    xs = []
    ys = []
    times = []

    for t in range(lookback - 1, len(X)):
        xs.append(X[t - lookback + 1:t + 1])
        ys.append(y_future[t])
        times.append(t)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(times, dtype=np.int64),
    )


class TimeWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def fit_scaler(X_train: np.ndarray):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def transform_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean[None, None, :]) / std[None, None, :]


class GRUDetector(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, h = self.rnn(x)
        last_hidden = h[-1]
        return self.head(last_hidden).squeeze(-1)


def get_pos_weight(y: np.ndarray, device: str):
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos < 1:
        return torch.tensor(1.0, device=device)
    return torch.tensor(neg / pos, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(yb)
        n += len(yb)

    return running_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(yb.numpy())

    return np.concatenate(probs), np.concatenate(labels)


def threshold_from_validation(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    I choose the alert threshold on validation rather than fixing 0.5.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    best_thr = 0.5
    best_f1 = -1.0

    for i, thr in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(np.int64)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true.astype(np.int64),
        y_pred,
        average="binary",
        zero_division=0,
    )

    out = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")

    return out


def keep_intervals_in_range(intervals, start_t, end_t):
    kept = []
    for s, e in intervals:
        if e < start_t or s > end_t:
            continue
        kept.append((s, e))
    return kept


def incident_metrics(y_prob, times, incidents, threshold, horizon, eval_start, eval_end):
    """
    Incident is counted as detected if there is at least one alert in [start-horizon, start].
    """
    alert_times = times[y_prob >= threshold]
    relevant = keep_intervals_in_range(incidents, eval_start, eval_end)

    detected = 0
    lead_times = []

    for start, end in relevant:
        lo = start - horizon
        hi = start
        hits = alert_times[(alert_times >= lo) & (alert_times <= hi)]
        if len(hits) > 0:
            detected += 1
            lead_times.append(int(start - hits.min()))

    region = np.zeros(eval_end - eval_start + 1, dtype=np.int64)
    for start, end in relevant:
        lo = max(eval_start, start - horizon)
        hi = min(eval_end, start)
        region[lo - eval_start:hi - eval_start + 1] = 1

    false_alerts = 0
    for t in alert_times:
        if eval_start <= t <= eval_end and region[t - eval_start] == 0:
            false_alerts += 1

    total_steps = eval_end - eval_start + 1
    n_alerts = ((alert_times >= eval_start) & (alert_times <= eval_end)).sum()

    return {
        "incident_recall": float(detected / len(relevant)) if len(relevant) > 0 else float("nan"),
        "detected_incidents": int(detected),
        "total_incidents": int(len(relevant)),
        "false_alerts": int(false_alerts),
        "alerts_per_1000_steps": float(1000.0 * n_alerts / max(total_steps, 1)),
        "mean_lead_time": float(np.mean(lead_times)) if len(lead_times) > 0 else float("nan"),
        "median_lead_time": float(np.median(lead_times)) if len(lead_times) > 0 else float("nan"),
    }


def plot_raw_series(X, incidents, feature_names, save_path):
    fig, axes = plt.subplots(X.shape[1], 1, figsize=(14, 10), sharex=True)

    for i in range(X.shape[1]):
        axes[i].plot(X[:, i], linewidth=1.0)
        axes[i].set_ylabel(feature_names[i])
        for s, e in incidents:
            axes[i].axvspan(s, e, color="red", alpha=0.15)

    axes[-1].set_xlabel("time step")
    fig.suptitle("Synthetic metrics and incident intervals")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_predictions(X, test_times, test_prob, threshold, incidents, feature_names, save_path):
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)

    for i in range(X.shape[1]):
        axes[i].plot(X[:, i], linewidth=1.0)
        axes[i].set_ylabel(feature_names[i])
        for s, e in incidents:
            axes[i].axvspan(s, e, color="red", alpha=0.12)

    axes[4].plot(test_times, test_prob, linewidth=1.5, label="predicted risk")
    axes[4].axhline(threshold, linestyle="--", linewidth=1.2, label=f"threshold={threshold:.3f}")

    alert_times = test_times[test_prob >= threshold]
    if len(alert_times) > 0:
        axes[4].scatter(alert_times, test_prob[test_prob >= threshold], s=12, label="alerts")

    for s, e in incidents:
        axes[4].axvspan(s, e, color="red", alpha=0.12)

    axes[4].set_ylabel("risk")
    axes[4].set_xlabel("time step")
    axes[4].legend()

    fig.suptitle("Predictions on test split")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    cfg = Settings()
    seed_everything(cfg.seed)
    make_dir(cfg.out_dir)

    print("device:", cfg.device)

    X_raw, incident_mask, incidents, feature_names = build_synthetic_series(cfg)
    y_future = make_future_labels(incident_mask, cfg.horizon)
    X_win, y_win, times = make_windows(X_raw, y_future, cfg.lookback)

    n_total = len(X_win)
    train_end = int(n_total * cfg.train_ratio)
    val_end = int(n_total * (cfg.train_ratio + cfg.val_ratio))

    X_train, y_train, t_train = X_win[:train_end], y_win[:train_end], times[:train_end]
    X_val, y_val, t_val = X_win[train_end:val_end], y_win[train_end:val_end], times[train_end:val_end]
    X_test, y_test, t_test = X_win[val_end:], y_win[val_end:], times[val_end:]

    mean, std = fit_scaler(X_train)
    X_train = transform_windows(X_train, mean, std)
    X_val = transform_windows(X_val, mean, std)
    X_test = transform_windows(X_test, mean, std)

    train_ds = TimeWindowDataset(X_train, y_train)
    val_ds = TimeWindowDataset(X_val, y_val)
    test_ds = TimeWindowDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GRUDetector(n_features=cfg.n_metrics, hidden_dim=cfg.hidden_dim).to(cfg.device)

    pos_weight = get_pos_weight(y_train, cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("train examples:", len(train_ds))
    print("val examples:  ", len(val_ds))
    print("test examples: ", len(test_ds))
    print("positive rate in train:", float(y_train.mean()))
    print("pos_weight:", float(pos_weight.detach().cpu()))

    best_state = None
    best_val_ap = -1.0
    best_epoch = -1
    patience_left = cfg.early_stop

    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_prob, val_true = predict(model, val_loader, cfg.device)
        val_ap = average_precision_score(val_true, val_prob)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_pr_auc": float(val_ap),
        })

        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_pr_auc={val_ap:.4f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop
        else:
            patience_left -= 1
            if patience_left == 0:
                print("stopping early")
                break

    print("best epoch:", best_epoch)
    print("best validation PR-AUC:", best_val_ap)

    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob, val_true = predict(model, val_loader, cfg.device)
    threshold = threshold_from_validation(val_true, val_prob)
    print("chosen threshold:", threshold)

    test_prob, test_true = predict(model, test_loader, cfg.device)

    val_metrics = classification_metrics(val_true, val_prob, threshold)
    test_metrics = classification_metrics(test_true, test_prob, threshold)

    print("\nvalidation metrics")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\ntest metrics")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    inc_metrics = incident_metrics(
        y_prob=test_prob,
        times=t_test,
        incidents=incidents,
        threshold=threshold,
        horizon=cfg.horizon,
        eval_start=int(t_test.min()),
        eval_end=int(t_test.max()),
    )

    print("\nincident-level test metrics")
    for k, v in inc_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    plot_raw_series(
        X=X_raw,
        incidents=incidents,
        feature_names=feature_names,
        save_path=os.path.join(cfg.out_dir, "synthetic_series.png"),
    )

    plot_predictions(
        X=X_raw,
        test_times=t_test,
        test_prob=test_prob,
        threshold=threshold,
        incidents=incidents,
        feature_names=feature_names,
        save_path=os.path.join(cfg.out_dir, "test_predictions.png"),
    )

    results = {
        "settings": cfg.__dict__,
        "best_epoch": best_epoch,
        "best_val_pr_auc": float(best_val_ap),
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "incident_test_metrics": inc_metrics,
        "incident_intervals": incidents,
        "history": history,
    }

    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "feature_names": feature_names,
            "settings": cfg.__dict__,
        },
        os.path.join(cfg.out_dir, "model.pt"),
    )

    print("\nsaved results in:", cfg.out_dir)


if __name__ == "__main__":
    main()