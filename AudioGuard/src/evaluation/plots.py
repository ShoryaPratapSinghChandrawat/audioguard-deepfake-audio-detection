"""Draw curves exclusively from saved JSON results, keeping plot numbers traceable."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(run):
    run = Path(run)
    folder = run / "figures"
    folder.mkdir(exist_ok=True)
    training = json.loads((run / "training.json").read_text(encoding="utf-8"))
    history = training.get("fit", {}).get("history", [])
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for part in ("train", "dev"):
            for ax, key in zip(axes, ("loss", "accuracy")):
                ax.plot([h["epoch"] for h in history], [h[part][key] for h in history], label=part)
                ax.set(xlabel="epoch", ylabel=key)
                ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "training.png")
        plt.close(fig)
    for path in sorted(run.glob("eval_*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("status") != "completed":
            continue
        fig, ax = plt.subplots()
        ax.plot(obj["roc"]["fpr"], obj["roc"]["tpr"], label="eval; frozen dev operating point")
        ax.scatter([obj["metrics"]["far"]], [1-obj["metrics"]["frr"]])
        ax.set(xlabel="False positive rate (bonafide flagged)", ylabel="Recall (spoof detected)")
        ax.legend()
        fig.savefig(folder / (path.stem + "_roc.png"))
        plt.close(fig)
        fig, ax = plt.subplots()
        matrix = np.asarray(obj["metrics"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["bonafide", "spoof"],
               yticklabels=["bonafide", "spoof"], xlabel="Predicted", ylabel="True")
        fig.savefig(folder / (path.stem + "_confusion.png"), bbox_inches="tight")
        plt.close(fig)
