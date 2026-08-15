"""
model.py
--------
The Machine Learning part of the project.

Steps implemented here (all with basic scikit-learn tools):
1. Train/test split with stratification
2. Two pipelines: Logistic Regression and Decision Tree
3. 5-fold Stratified cross-validation on the training set
4. Evaluation on the untouched test set
5. Automatic selection of the best model using the F1-score
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


def get_models():
    """The two basic classifiers required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    }


def split_data(X, y):
    """
    80/20 train-test split.

    stratify=y keeps the same arrest / no-arrest proportion in both parts,
    which matters because the target is imbalanced (~1/3 arrests).
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_pipeline(model, categorical, numerical):
    """Preprocessing + model in a single object, so no step can be forgotten."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical, numerical)),
            ("classifier", model),
        ]
    )


def run_cross_validation(pipeline, X_train, y_train, n_splits=N_SPLITS):
    """
    Stratified k-fold cross-validation on the TRAINING data only.

    Returns the mean and standard deviation of accuracy, precision, recall and F1.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1"]

    results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    summary = {}
    for metric in scoring:
        scores = results[f"test_{metric}"]
        summary[metric] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    return summary


def evaluate_on_test(pipeline, X_test, y_test):
    """Metrics on the held-out test set, including the confusion matrix."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # ROC-AUC is a useful extra check, but it is not our main metric.
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = cm
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = int(tn), int(fp), int(fn), int(tp)
    return metrics


def train_and_evaluate(X, y, categorical, numerical):
    """
    Full ML workflow. Returns one dictionary containing everything the
    Streamlit pages need to display.
    """
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipeline = build_pipeline(model, categorical, numerical)

        # 1) Cross-validation on the training set
        cv_summary = run_cross_validation(pipeline, X_train, y_train)

        # 2) Fit on the whole training set and evaluate on the test set
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_on_test(pipeline, X_test, y_test)

        results[name] = {"cv": cv_summary, "test": test_metrics}
        fitted_pipelines[name] = pipeline

    best_name = select_best_model(results)

    return {
        "results": results,
        "pipelines": fitted_pipelines,
        "best_model_name": best_name,
        "best_pipeline": fitted_pipelines[best_name],
        "explanation": build_explanation(results, best_name),
        "split_info": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_arrest_rate": float(y_train.mean() * 100),
            "test_arrest_rate": float(y_test.mean() * 100),
        },
        "X_train": X_train,
    }


def select_best_model(results):
    """
    Pick the model with the highest test F1-score.

    F1 is used because it balances precision and recall, which is the right
    choice for an imbalanced target - a model could reach ~67% accuracy simply
    by predicting 'No Arrest' every time.
    """
    return max(results, key=lambda name: results[name]["test"]["f1"])


def build_explanation(results, best_name):
    """Write the selection sentence using the numbers that were actually measured."""
    best = results[best_name]["test"]
    others = [n for n in results if n != best_name]

    text = (
        f"**{best_name}** was selected because it achieved the highest F1-score "
        f"({best['f1']:.3f}) on the test set, with an accuracy of {best['accuracy']:.3f}, "
        f"precision of {best['precision']:.3f} and recall of {best['recall']:.3f}."
    )

    if others:
        other = others[0]
        other_metrics = results[other]["test"]
        gap = best["f1"] - other_metrics["f1"]
        text += (
            f" The {other} reached an F1-score of {other_metrics['f1']:.3f} "
            f"({gap:+.3f} difference), accuracy {other_metrics['accuracy']:.3f}, "
            f"precision {other_metrics['precision']:.3f} and recall {other_metrics['recall']:.3f}."
        )

    text += (
        f" Precision of {best['precision']:.3f} means that when the model predicts an arrest it is "
        f"correct about {best['precision'] * 100:.0f}% of the time, while a recall of {best['recall']:.3f} "
        f"means it finds roughly {best['recall'] * 100:.0f}% of all the cases that really ended in an arrest."
    )
    return text


# ---------------------------------------------------------------------------
# Small helpers that turn the result dictionaries into display tables
# ---------------------------------------------------------------------------
def cv_results_table(results):
    """Cross-validation table: mean +/- standard deviation for each metric."""
    rows = []
    for name, res in results.items():
        cv = res["cv"]
        rows.append(
            {
                "Model": name,
                "CV Accuracy": f"{cv['accuracy']['mean']:.4f} +/- {cv['accuracy']['std']:.4f}",
                "CV Precision": f"{cv['precision']['mean']:.4f} +/- {cv['precision']['std']:.4f}",
                "CV Recall": f"{cv['recall']['mean']:.4f} +/- {cv['recall']['std']:.4f}",
                "CV F1": f"{cv['f1']['mean']:.4f} +/- {cv['f1']['std']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def test_results_table(results):
    """Test-set table with one row per model."""
    rows = []
    for name, res in results.items():
        test = res["test"]
        row = {
            "Model": name,
            "Accuracy": round(test["accuracy"], 4),
            "Precision": round(test["precision"], 4),
            "Recall": round(test["recall"], 4),
            "F1 Score": round(test["f1"], 4),
        }
        if "roc_auc" in test:
            row["ROC-AUC"] = round(test["roc_auc"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_long_table(results):
    """Melted table (Model, Metric, Score) used for the grouped bar chart."""
    rows = []
    for name, res in results.items():
        test = res["test"]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            rows.append({"Model": name, "Metric": metric.capitalize(),
                         "Score": round(test[metric], 4)})
    return pd.DataFrame(rows)


def predict_single(pipeline, input_dict, feature_order):
    """
    Predict one crime record entered in the Streamlit form.

    The input is turned into a one-row DataFrame with exactly the same columns
    as the training data, so it flows through the identical preprocessing steps.
    """
    row = pd.DataFrame([input_dict])[feature_order]

    prediction = int(pipeline.predict(row)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(row)[0][1])

    return prediction, probability
