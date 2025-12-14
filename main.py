from pathlib import Path
from src.data_understanding import summarize_dataset
from src.data_preparation import prepare_datasets
from src.modeling import train_model
from src.evaluation import evaluate


def run_pipeline():
    root = Path(__file__).resolve().parents[1]
    cfg_path = str(root / "configs" / "config.yaml")

    print("[CRISP-DM] Data Understanding...")
    du = summarize_dataset(cfg_path)
    print(du)

    print("[CRISP-DM] Data Preparation...")
    dp = prepare_datasets(cfg_path)
    print(dp)

    print("[CRISP-DM] Modeling...")
    md = train_model(cfg_path)
    print(md)

    print("[CRISP-DM] Evaluation...")
    ev = evaluate(cfg_path)
    print(ev)


if __name__ == "__main__":
    run_pipeline()
