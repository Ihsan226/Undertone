from pathlib import Path
import yaml
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    models_dir = paths["models_dir"]
    reports_dir = paths["reports_dir"]

    clf = load(Path(models_dir) / "model.joblib")
    X_test, y_test = load(Path(models_dir) / "test_data.joblib")
    paths_test_path = Path(models_dir) / "test_paths.joblib"
    paths_test = load(paths_test_path) if paths_test_path.exists() else None

    y_pred = clf.predict(X_test)
    # Optionally get probabilities or decision scores
    proba = None
    scores = None
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_test)
        except Exception:
            proba = None
    if proba is None and hasattr(clf, "decision_function"):
        try:
            scores = clf.decision_function(X_test)
        except Exception:
            scores = None
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = Path(reports_dir) / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # Save text report
    report_path = Path(reports_dir) / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save misclassifications if paths are available
    mis_path = None
    if paths_test is not None:
        rows = []
        # Determine class labels order from classifier if available
        classes = getattr(clf, "classes_", None)
        for idx, (p, t, pr) in enumerate(zip(paths_test, y_test, y_pred)):
            if t != pr:
                entry = {"path": p, "true": t, "pred": pr}
                # Add probability/score details when available
                if proba is not None:
                    entry["pred_confidence"] = float(proba[idx][classes.tolist().index(pr)]) if classes is not None else float(max(proba[idx]))
                    if classes is not None and t in classes.tolist():
                        entry["true_class_prob"] = float(proba[idx][classes.tolist().index(t)])
                elif scores is not None:
                    # For decision function, store raw scores and the selected score
                    if hasattr(scores, "shape") and len(getattr(scores, "shape", [])):
                        if classes is not None:
                            entry["pred_score"] = float(scores[idx][classes.tolist().index(pr)])
                            if t in classes.tolist():
                                entry["true_class_score"] = float(scores[idx][classes.tolist().index(t)])
                        else:
                            # Binary case may be 1D
                            val = scores[idx]
                            entry["pred_score"] = float(val if not isinstance(val, (list, tuple)) else val[0])
                rows.append(entry)
        mis_df = pd.DataFrame(rows)
        mis_path = Path(reports_dir) / "misclassifications.csv"
        mis_df.to_csv(mis_path, index=False)

    out = {"confusion_matrix": str(cm_path), "report": str(report_path)}
    if mis_path:
        out["misclassifications"] = str(mis_path)
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = str(root / "configs" / "config.yaml")
    info = evaluate(cfg_path)
    print(info)
