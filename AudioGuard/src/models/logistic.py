"""Seeded minibatch logistic regression with train-only standardization.

NumPy is the portable CPU backend. If the optional pinned PyTorch runtime can
use CUDA, identical gradient equations run there. There is no eval-based fitting.
"""
import numpy as np
from scipy.special import expit


def select_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but usable PyTorch CUDA runtime is absent")
    return "cuda" if available and requested != "cpu" else "cpu"


def fit(x, y, cfg, seed):
    device = select_device(cfg["device"])
    print("Selected device:", device)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    w, bias = np.zeros(z.shape[1]), 0.0
    rng = np.random.default_rng(seed)
    history = []
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["learning_rate"] <= 0 or cfg["l2"] < 0:
        raise ValueError("Invalid optimizer configuration")
    if device == "cuda":
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        tz = torch.tensor(z, dtype=torch.float64, device=device)
        ty = torch.tensor(y, dtype=torch.float64, device=device)
        tw = torch.zeros(z.shape[1], dtype=torch.float64, device=device)
        tb = torch.zeros((), dtype=torch.float64, device=device)
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(y))
        for start in range(0, len(y), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            if device == "cuda":
                ti = torch.tensor(idx, device=device)
                xb, yb = tz[ti], ty[ti]
                err = torch.sigmoid(xb @ tw + tb) - yb
                tw -= cfg["learning_rate"] * (xb.T @ err / len(idx) + cfg["l2"] * tw)
                tb -= cfg["learning_rate"] * err.mean()
            else:
                xb, yb = z[idx], y[idx]
                err = expit(xb @ w + bias) - yb
                w -= cfg["learning_rate"] * (xb.T @ err / len(idx) + cfg["l2"] * w)
                bias -= cfg["learning_rate"] * err.mean()
        if device == "cuda":
            w, bias = tw.cpu().numpy(), float(tb.cpu())
        logits = z @ w + bias
        # logaddexp is stable even for extreme logits.
        loss = np.mean(np.logaddexp(0, logits) - y * logits) + cfg["l2"] * np.dot(w, w) / 2
        history.append({"epoch": epoch + 1, "training_objective": float(loss)})
    return {"mean": mean, "std": std, "weights": w, "bias": np.array(bias)}, history, device


def score(x, model):
    return expit(((x - model["mean"]) / model["std"]) @ model["weights"] + model["bias"])
