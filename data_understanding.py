import os
from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import list_images_by_label, LABELS, ensure_dirs


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def summarize_dataset(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    reports_dir = paths["reports_dir"]
    ensure_dirs([reports_dir])

    train_dir = paths["train_dir"]
    exts = tuple(cfg["images"]["extensions"])

    items = list_images_by_label(train_dir, exts)
    df = pd.DataFrame(items, columns=["path", "label"])  # noqa: PD901

    # Counts per label
    counts = Counter(df["label"]) if not df.empty else Counter()

    # Save summary CSV
    summary_csv = Path(reports_dir) / "data_summary.csv"
    df.to_csv(summary_csv, index=False)

    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.title("Training Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plot_path = Path(reports_dir) / "class_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "num_images": len(df),
        "counts": counts,
        "summary_csv": str(summary_csv),
        "class_plot": str(plot_path),
        "labels": LABELS,
    }


if __name__ == "__main__":
    # For quick manual run
    root = Path(__file__).resolve().parents[1]
    cfg_path = str(root / "configs" / "config.yaml")
    info = summarize_dataset(cfg_path)
    print(info)
