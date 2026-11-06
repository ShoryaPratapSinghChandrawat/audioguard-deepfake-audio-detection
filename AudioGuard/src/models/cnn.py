"""Small configurable spectrogram CNN and deterministic train/dev fitting.

The sanity check deliberately memorizes train-only trials, then discards its
weights. Normal training starts fresh and selects checkpoints using dev loss only.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.features.acoustic import cached
from src.evaluation.metrics import empirical_eer


class SmallCNN(nn.Module):
    """Conv/BatchNorm/ReLU/MaxPool blocks, global pooling and a single spoof logit."""
    def __init__(self, cfg):
        super().__init__()
        blocks, previous = [], 1
        for channels in cfg["channels"]:
            blocks.extend([nn.Conv2d(previous, channels, 3, padding=1), nn.BatchNorm2d(channels),
                           nn.ReLU(), nn.MaxPool2d(2)])
            previous = channels
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(cfg["dropout"]), nn.Linear(previous, 1))

    def forward(self, value):
        return self.head(self.body(value)).squeeze(1)


def seed_torch(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def train_scaler(rows, cfg, kind):
    total, square, frames = None, None, 0
    for row in rows:
        x = cached(row["path"], cfg, kind).astype(np.float64)
        if total is None:
            total, square = np.zeros(x.shape[0]), np.zeros(x.shape[0])
        total += x.sum(axis=1)
        square += (x*x).sum(axis=1)
        frames += x.shape[1]
    mean = total / frames
    std = np.sqrt(np.maximum(square / frames - mean*mean, 0))
    return {"mean": mean.astype(np.float32), "std": np.where(std > 1e-6, std, 1).astype(np.float32)}


def tensor_input(x, scaler, frames):
    x = (x - scaler["mean"][:, None]) / scaler["std"][:, None]
    if x.shape[1] > frames:
        start = (x.shape[1] - frames) // 2
        x = x[:, start:start+frames]
    else:
        # Zero after standardization represents the TRAIN mean, not waveform silence.
        x = np.pad(x, ((0, 0), (0, frames-x.shape[1])))
    return torch.tensor(x[None], dtype=torch.float32)


class Clips(Dataset):
    """Lazy cached matrices, with identical deterministic crop/pad across all splits."""
    def __init__(self, rows, features, kind, scaler, frames):
        self.rows, self.features, self.kind = rows, features, kind
        self.scaler, self.frames = scaler, frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        value = tensor_input(cached(row["path"], self.features, self.kind), self.scaler, self.frames)
        return value, torch.tensor(row["label"], dtype=torch.float32)


def make_loader(rows, features, kind, scaler, cfg, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Clips(rows, features, kind, scaler, cfg["frames"]),
                      batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], generator=generator)


def epoch(model, loader, loss_fn, device, optimizer=None, return_scores=False):
    model.train(optimizer is not None)
    total, correct, count = 0., 0, 0
    scores, labels = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(y)
            correct += int(((logits >= 0) == (y == 1)).sum())
            count += len(y)
            if return_scores:
                scores.extend(logits.detach().cpu().tolist())
                labels.extend(y.detach().cpu().tolist())
    result = {"loss": total/count, "accuracy": correct/count, "n": count}
    return (result, (np.asarray(labels), np.asarray(scores))) if return_scores else result


def balanced_sanity_rows(rows, count, seed):
    if count < 2 or count % 2:
        raise ValueError("Sanity subset size must be positive, even and contain both labels")
    if any(row.get("partition") != "train" for row in rows):
        raise ValueError("Overfit sanity check accepts only official train-partition rows")
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = [row for row in rows if row["label"] == label]
        if len(candidates) < count // 2:
            raise ValueError("Insufficient unique train trials for sanity check; never duplicate or use dev/eval")
        selected.extend([candidates[i] for i in rng.choice(len(candidates), count // 2, replace=False)])
    return selected


def overfit(rows, features, kind, scaler, cfg, seed, device):
    subset = balanced_sanity_rows(rows, cfg["sanity"]["samples"], seed)
    seed_torch(seed, cfg["cpu_threads"])
    model = SmallCNN(cfg).to(device)
    # Disable dropout only in the diagnostic to make memorization easier to interpret.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    loader = make_loader(subset, features, kind, scaler, cfg, True, seed)
    diagnostic = make_loader(subset, features, kind, scaler, cfg)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1., device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["sanity"]["learning_rate"], weight_decay=0)
    history, passed = [], False
    for index in range(cfg["sanity"]["epochs"]):
        epoch(model, loader, loss_fn, device, optimizer)
        result = {"epoch": index+1, **epoch(model, diagnostic, loss_fn, device)}
        history.append(result)
        passed = result["accuracy"] >= cfg["sanity"]["minimum_accuracy"] and result["loss"] <= cfg["sanity"]["maximum_loss"]
        if passed:
            break
    return {"status": "passed" if passed else "failed", "train_only": True,
            "samples": len(subset), "utterances": [r["utterance"] for r in subset], "history": history}


def fit(train, dev, features, kind, scaler, cfg, seed, device):
    seed_torch(seed, cfg["cpu_threads"])
    if cfg["epochs"] < 1 or cfg["patience"] < 1:
        raise ValueError("epochs and patience must be positive")
    model = SmallCNN(cfg).to(device)
    positive = sum(r["label"] for r in train)
    pos_weight = (len(train)-positive) / positive
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    training = make_loader(train, features, kind, scaler, cfg, True, seed)
    validation = make_loader(dev, features, kind, scaler, cfg)
    history, best, stale, state, best_epoch = [], float("inf"), 0, None, None
    for index in range(cfg["epochs"]):
        tr = epoch(model, training, loss_fn, device, optimizer)
        dv, (dev_labels, dev_scores) = epoch(model, validation, loss_fn, device, return_scores=True)
        dev_eer = empirical_eer(dev_labels, dev_scores)["eer"]
        history.append({"epoch": index+1, "train": tr, "dev": dv, "dev_eer": dev_eer})
        print(f"Epoch {index+1}/{cfg['epochs']}: train_loss={tr['loss']:.8f} "
              f"dev_loss={dv['loss']:.8f} dev_eer={dev_eer:.8f}", flush=True)
        if dv["loss"] < best - cfg["min_delta"]:
            best, stale, best_epoch = dv["loss"], 0, index+1
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(state)
    return model, {"history": history, "best_epoch": best_epoch, "pos_weight": pos_weight,
                   "checkpoint_selection": "minimum dev weighted BCE loss"}


def predict(model, rows, features, kind, scaler, cfg, device):
    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in make_loader(rows, features, kind, scaler, cfg):
            values.extend(model(x.to(device)).cpu().tolist())
    return np.asarray(values)
