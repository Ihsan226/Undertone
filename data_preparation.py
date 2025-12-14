from pathlib import Path
import yaml
import numpy as np
from skimage import io, color, transform, feature
from sklearn.model_selection import train_test_split
from joblib import dump
from tqdm import tqdm

from .utils import list_images_by_label, ensure_dirs


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_preprocess(image_path: str, target_size=(128, 128), normalize=True):
    img = io.imread(image_path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    img = transform.resize(img, target_size, anti_aliasing=True)
    if normalize:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def extract_features(img, mode: str = "hog"):
    if mode == "raw":
        return img.reshape(-1)
    # HOG on grayscale
    gray = color.rgb2gray(img)
    hog_vec = feature.hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
    )
    return hog_vec


def prepare_datasets(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    exts = tuple(cfg["images"]["extensions"])
    target_size = tuple(cfg["images"]["target_size"])
    normalize = bool(cfg["images"]["normalize"])
    features = cfg["model"]["features"]

    data_dir = paths["train_dir"]
    models_dir = paths["models_dir"]
    reports_dir = paths["reports_dir"]
    ensure_dirs([models_dir, reports_dir])

    items = list_images_by_label(data_dir, exts)
    if not items:
        raise RuntimeError("No images found in train directory")

    X = []
    y = []
    paths = []
    for p, label in tqdm(items, desc="Preprocessing images"):
        img = load_and_preprocess(p, target_size=target_size, normalize=normalize)
        vec = extract_features(img, mode=features)
        X.append(vec)
        y.append(label)
        paths.append(p)
    X = np.array(X)

    # Split into train/val/test
    val_size = float(cfg["prep"]["val_size"])  # portion of remaining after test
    test_size = float(cfg["prep"]["test_size"])  # portion of total
    random_state = int(cfg["prep"]["random_state"])

    # First, hold out test set from the full dataset
    X_train_full, X_test, y_train_full, y_test, paths_train_full, paths_test = train_test_split(
        X, y, paths, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then, split the remaining training set into train and validation
    # Here, val_size is the portion of the remaining (post-test) data
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
        X_train_full, y_train_full, paths_train_full, test_size=val_size, stratify=y_train_full, random_state=random_state
    )

    dump((X_train, y_train), Path(models_dir) / "train_data.joblib")
    dump((X_val, y_val), Path(models_dir) / "val_data.joblib")
    dump((X_test, y_test), Path(models_dir) / "test_data.joblib")
    # Save corresponding image paths for downstream error analysis
    dump(paths_train, Path(models_dir) / "train_paths.joblib")
    dump(paths_val, Path(models_dir) / "val_paths.joblib")
    dump(paths_test, Path(models_dir) / "test_paths.joblib")

    return {
        "train": (len(X_train), len(y_train)),
        "val": (len(X_val), len(y_val)),
        "test": (len(X_test), len(y_test)),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = str(root / "configs" / "config.yaml")
    info = prepare_datasets(cfg_path)
    print(info)
