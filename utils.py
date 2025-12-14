import os
from pathlib import Path
from typing import List, Tuple

LABELS = ["Black", "Brown", "White"]


def list_images_by_label(train_dir: str, extensions: Tuple[str, ...]) -> List[Tuple[str, str]]:
    items = []
    for label in LABELS:
        label_dir = Path(train_dir) / label
        if not label_dir.exists():
            continue
        for name in os.listdir(label_dir):
            p = label_dir / name
            if p.suffix.lower() in extensions:
                items.append((str(p), label))
    return items


def ensure_dirs(paths: List[str]):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
