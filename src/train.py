"""Train and compare seven classifiers for breast-cancer metastasis/recurrence.

Leakage-safe protocol:
  * stratified 70/30 train/test split, made once (seed 101)
  * imputation, scaling and SMOTE-ENN resampling are steps of an imblearn
    Pipeline, so inside cross-validation they are fitted on training folds only
  * hyper-parameters tuned by 5-fold stratified CV on ROC AUC
  * the test set is used once, at the end

Usage:  python src/train.py --data /path/to/BreastCancerData.sav
Writes only aggregate results to results/ and figures/ (no patient-level data).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prepare import CATEGORICAL_TO_IMPUTE, FEATURES, load_dataset

warnings.filterwarnings("ignore")
SEED = 101

MODELS = {
    "Logistic regression": (LogisticRegression(max_iter=2000),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "LDA": (LinearDiscriminantAnalysis(),
            {"clf__solver": ["svd"]}),
    "Decision tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 20, 50]}),
    "Bagging": (BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=SEED),
                {"clf__estimator__min_samples_leaf": [1, 5, 20]}),
    "Random forest": (RandomForestClassifier(n_estimators=300, random_state=SEED),
                      {"clf__max_features": ["sqrt", 0.3], "clf__min_samples_leaf": [1, 5, 20]}),
    "SVM (RBF)": (SVC(kernel="rbf", probability=True, random_state=SEED),
                  {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}),
    "KNN": (KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 15, 31, 51]}),
}


def make_pipeline(model) -> Pipeline:
    numeric = [c for c in FEATURES if c not in CATEGORICAL_TO_IMPUTE]
    impute = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", SimpleImputer(strategy="most_frequent"), CATEGORICAL_TO_IMPUTE),
    ])
    return Pipeline([("impute", impute), ("scale", StandardScaler()),
                     ("resample", SMOTEENN(random_state=SEED)), ("clf", model)])


def auc_ci(y_true, prob, n_boot=2000, seed=SEED) -> tuple[float, float]:
    """95% bootstrap interval for the test AUC (the test set has few events)."""
    rng = np.random.default_rng(seed)
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() != y_true[idx].max():
            aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def evaluate(y_true, prob, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    lo, hi = auc_ci(y_true, prob)
    return {"AUC": roc_auc_score(y_true, prob), "AUC_95CI_low": lo, "AUC_95CI_high": hi,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "accuracy": (tp + tn) / len(y_true)}


def main(data_path: str) -> None:
    X, y = load_dataset(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                              random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, rocs = [], {}
    for name, (model, grid) in MODELS.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        prob = search.predict_proba(X_te)[:, 1]
        rows.append({"model": name, "cv_AUC": search.best_score_,
                     **evaluate(y_te, prob),
                     "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()}})
        rocs[name] = roc_curve(y_te, prob)
        print(f"{name:20s} CV AUC {search.best_score_:.3f}  test AUC {rows[-1]['AUC']:.3f}")

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    res = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    res.round(3).to_csv("results/model_comparison.csv", index=False)
    pd.Series({"n_patients": len(y), "n_events": int(y.sum()),
               "event_rate": round(y.mean(), 4), "n_train": len(y_tr), "n_test": len(y_te),
               "n_features": X.shape[1]}).to_csv("results/dataset_summary.csv", header=["value"])

    # ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (fpr, tpr, _) in rocs.items():
        auc = res.set_index("model").loc[name, "AUC"]
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="Test-set ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig("figures/roc_curves.png", dpi=130); plt.close(fig)

    print(res.drop(columns="best_params").round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the confidential .sav export")
    main(ap.parse_args().data)
