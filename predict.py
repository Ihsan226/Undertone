from pathlib import Path
import argparse
import yaml
import numpy as np
from joblib import load
from skimage import io, color, transform, feature


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(image_path: str, target_size=(128, 128), normalize=True, features="hog"):
    img = io.imread(image_path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    img = transform.resize(img, target_size, anti_aliasing=True)
    if normalize:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if features == "raw":
        vec = img.reshape(-1)
    else:
        gray = color.rgb2gray(img)
        vec = feature.hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm="L2-Hys",
        )
    return vec


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_path = str(root / "configs" / "config.yaml")
    cfg = load_config(cfg_path)

    parser = argparse.ArgumentParser(description="Predict class of an image")
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    model_path = Path(cfg["paths"]["models_dir"]) / "model.joblib"
    clf = load(model_path)

    vec = preprocess(
        str(img_path),
        target_size=tuple(cfg["images"]["target_size"]),
        normalize=bool(cfg["images"]["normalize"]),
        features=cfg["model"]["features"],
    )
    pred = clf.predict([vec])[0]
    print(pred)


if __name__ == "__main__":
    main()
