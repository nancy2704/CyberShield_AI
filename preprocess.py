"""
preprocess.py
-------------
Handles synthetic dataset generation and feature preprocessing
for CyberShield AI – AI-Based Threat Detection System.

NOTE: This project uses SYNTHETIC (artificially generated) data for
demonstration purposes only. It does NOT perform real-time monitoring
and cannot guarantee detection of actual cyberattacks.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Feature columns expected by the model ────────────────────────────────────
FEATURE_COLUMNS = [
    "duration",          # Connection duration in seconds
    "src_bytes",         # Bytes sent from source
    "dst_bytes",         # Bytes sent to destination
    "packet_count",      # Total packets exchanged
    "protocol_type",     # Protocol: tcp / udp / icmp
    "flag",              # Connection flag: SF / REJ / S0 / RSTO
    "same_srv_rate",     # Fraction of connections to the same service
    "diff_srv_rate",     # Fraction of connections to different services
    "srv_count",         # Number of connections to the same service (last 2 s)
]


def generate_synthetic_dataset(n_samples: int = 2000, save_path: str = "data/network_traffic.csv") -> pd.DataFrame:
    """
    Generate a labelled synthetic network-traffic dataset and save it to CSV.

    Classes
    -------
    0 – Normal traffic
    1 – Suspicious traffic

    Returns the generated DataFrame.
    """
    n_normal = int(n_samples * 0.65)
    n_suspicious = n_samples - n_normal

    protocols = ["tcp", "udp", "icmp"]
    flags_normal = ["SF", "SF", "SF", "RSTO"]          # mostly SF for normal
    flags_suspicious = ["REJ", "S0", "REJ", "S0", "SF"]  # more REJ / S0 for suspicious

    # ── Normal traffic ────────────────────────────────────────────────────────
    normal = pd.DataFrame({
        "duration":       np.random.exponential(scale=5, size=n_normal).clip(0, 60),
        "src_bytes":      np.random.randint(100, 5_000, size=n_normal),
        "dst_bytes":      np.random.randint(100, 8_000, size=n_normal),
        "packet_count":   np.random.randint(5, 100, size=n_normal),
        "protocol_type":  np.random.choice(protocols, size=n_normal, p=[0.6, 0.3, 0.1]),
        "flag":           np.random.choice(flags_normal, size=n_normal),
        "same_srv_rate":  np.random.uniform(0.6, 1.0, size=n_normal),
        "diff_srv_rate":  np.random.uniform(0.0, 0.2, size=n_normal),
        "srv_count":      np.random.randint(1, 20, size=n_normal),
        "label":          0,
    })

    # ── Suspicious traffic ────────────────────────────────────────────────────
    suspicious = pd.DataFrame({
        "duration":       np.random.exponential(scale=0.5, size=n_suspicious).clip(0, 5),
        "src_bytes":      np.random.randint(0, 500, size=n_suspicious),
        "dst_bytes":      np.random.randint(0, 200, size=n_suspicious),
        "packet_count":   np.random.randint(100, 1_000, size=n_suspicious),
        "protocol_type":  np.random.choice(protocols, size=n_suspicious, p=[0.2, 0.2, 0.6]),
        "flag":           np.random.choice(flags_suspicious, size=n_suspicious),
        "same_srv_rate":  np.random.uniform(0.0, 0.3, size=n_suspicious),
        "diff_srv_rate":  np.random.uniform(0.5, 1.0, size=n_suspicious),
        "srv_count":      np.random.randint(200, 512, size=n_suspicious),
        "label":          1,
    })

    df = pd.concat([normal, suspicious], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    df.to_csv(save_path, index=False)
    print(f"[preprocess] Dataset saved -> {save_path}  ({len(df)} rows)")
    return df


def preprocess_features(df: pd.DataFrame):
    """
    Encode categorical columns and scale numeric features.

    Returns
    -------
    X_scaled : np.ndarray  – scaled feature matrix
    y        : pd.Series   – label column (if present, else None)
    scaler   : StandardScaler
    encoders : dict[str, LabelEncoder]
    """
    df = df.copy()

    # Encode categoricals
    encoders = {}
    for col in ["protocol_type", "flag"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    y = df["label"] if "label" in df.columns else None
    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders
