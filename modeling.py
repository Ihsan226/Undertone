from pathlib import Path
import yaml
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    models_dir = paths["models_dir"]

    X_train, y_train = load(Path(models_dir) / "train_data.joblib")
    X_val, y_val = load(Path(models_dir) / "val_data.joblib")

    reports_dir = cfg["paths"]["reports_dir"]

    # SVM baseline or tuned via GridSearchCV
    params = cfg["model"]["svm"]
    base_clf = SVC(
        C=params["C"], kernel=params["kernel"], gamma=params["gamma"], probability=params.get("probability", False)
    )

    if cfg["model"].get("tuning", {}).get("enabled", False):
        grid = cfg["model"].get("svm_grid", {})
        cv = int(cfg["model"].get("tuning", {}).get("cv", 3))
        search = GridSearchCV(base_clf, param_grid=grid, cv=cv, n_jobs=-1, scoring="f1_macro")
        search.fit(X_train, y_train)
        clf = search.best_estimator_
        # Save grid results
        results_df = pd.DataFrame(search.cv_results_)
        results_path = Path(reports_dir) / "svm_grid_results.csv"
        results_df.to_csv(results_path, index=False)
    else:
        clf = base_clf
        clf.fit(X_train, y_train)

    # Quick val report
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)

    dump(clf, Path(models_dir) / "model.joblib")
    dump(report, Path(models_dir) / "val_report.joblib")

    return {"trained": True, "model_path": str(Path(models_dir) / "model.joblib")}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = str(root / "configs" / "config.yaml")
    info = train_model(cfg_path)
    print(info)
