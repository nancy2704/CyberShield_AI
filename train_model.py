"""
train_model.py
--------------
Trains a Random Forest classifier on the synthetic network-traffic
dataset and saves the model artefacts for use by the Streamlit app.

Run this script once before launching the dashboard:
    python train_model.py

Output files
    models/rf_model.pkl   – trained RandomForestClassifier
    models/scaler.pkl     – fitted StandardScaler
    models/encoders.pkl   – fitted LabelEncoders (dict)
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess import generate_synthetic_dataset, preprocess_features

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/network_traffic.csv"
MODEL_PATH   = "models/rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/encoders.pkl"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_ESTIMATORS = 100
RANDOM_SEED  = 42
TEST_SIZE    = 0.2


def train():
    # 1. Generate (or load) dataset ───────────────────────────────────────────
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[train] Loaded existing dataset from {DATA_PATH}")
    else:
        print("[train] Dataset not found - generating a new one ...")
        df = generate_synthetic_dataset(n_samples=2000, save_path=DATA_PATH)

    # 2. Preprocess ───────────────────────────────────────────────────────────
    X, y, scaler, encoders = preprocess_features(df)

    # 3. Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[train] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 4. Train model ──────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("[train] Model training complete.")

    # 5. Evaluate ─────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Normal", "Suspicious"]
        ),
    }

    print("\n-- Evaluation Results ------------------------------------------")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-Score  : {metrics['f1_score']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    # 6. Save artefacts ───────────────────────────────────────────────────────
    joblib.dump(clf,      MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"\n[train] Artefacts saved -> {MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}")

    return clf, scaler, encoders, metrics


if __name__ == "__main__":
    train()
