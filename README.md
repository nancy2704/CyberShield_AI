# 🛡️ CyberShield AI — AI-Based Threat Detection System

> **IBM Internship Project** | Built with Python, Scikit-learn, Pandas, Streamlit and Plotly

---

## Overview

**CyberShield AI** is an educational machine-learning project that classifies simulated network traffic records as **Normal** or **Suspicious** using a Random Forest classifier. The interactive Streamlit dashboard provides threat summaries, traffic charts, model evaluation metrics and a CSV upload feature for custom analysis.

> ⚠️ **Disclaimer:** This project uses **synthetic (artificially generated) data** for demonstration purposes only. It does **not** perform real-time network monitoring and **cannot** guarantee detection of actual cyberattacks. It is intended solely for learning and internship demonstration.

---

## Features

- ✅ Synthetic labelled dataset generation (2,000 records)
- ✅ Random Forest classifier (Scikit-learn)
- ✅ Model evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ Interactive Streamlit dashboard with 4 pages
- ✅ CSV upload for custom traffic analysis
- ✅ Threat alerts, colour-coded predictions and downloadable results
- ✅ Feature importance chart

---

## Project Structure

```
CyberShield_AI/
├── app.py                  ← Streamlit dashboard
├── train_model.py          ← Model training & evaluation
├── preprocess.py           ← Dataset generation & preprocessing
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── data/
│   └── network_traffic.csv ← Auto-generated synthetic dataset
└── models/
    ├── rf_model.pkl        ← Trained Random Forest model
    ├── scaler.pkl          ← Fitted StandardScaler
    └── encoders.pkl        ← Fitted LabelEncoders
```

---

## Quick Start (Windows)

### 1. Prerequisites

- Python 3.9 or higher — download from https://www.python.org/downloads/
- During installation, check **"Add Python to PATH"**

### 2. Install dependencies

Open **Command Prompt** or **PowerShell** in the project folder and run:

```powershell
pip install -r requirements.txt
```

### 3. Train the model

```powershell
python train_model.py
```

This will:
- Generate `data/network_traffic.csv` (2,000 synthetic records)
- Train the Random Forest classifier
- Save model artefacts to the `models/` folder
- Print evaluation metrics to the console

### 4. Launch the dashboard

```powershell
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | Overview of the synthetic dataset with charts and threat summary |
| 🔍 Analyze Traffic | Upload your own CSV and get instant predictions |
| 📊 Model Evaluation | Confusion matrix, classification report, feature importances |
| ℹ️ About | How the system works, tech stack, limitations |

---

## CSV Upload Format

Your file must contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `duration` | Connection duration (seconds) | `2.5` |
| `src_bytes` | Bytes sent from source | `500` |
| `dst_bytes` | Bytes sent to destination | `1200` |
| `packet_count` | Total packets exchanged | `15` |
| `protocol_type` | Protocol (`tcp` / `udp` / `icmp`) | `tcp` |
| `flag` | Connection flag (`SF` / `REJ` / `S0` / `RSTO`) | `SF` |
| `same_srv_rate` | Fraction of connections to same service (0–1) | `0.85` |
| `diff_srv_rate` | Fraction of connections to different services (0–1) | `0.05` |
| `srv_count` | Number of connections to same service (last 2 s) | `8` |

A **sample CSV** can be downloaded directly from the *Analyze Traffic* page.

---

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Dashboard | Streamlit | 1.35 |
| ML Model | Scikit-learn | 1.5 |
| Data processing | Pandas / NumPy | 2.2 / 1.26 |
| Visualisation | Plotly | 5.22 |
| Model persistence | Joblib | 1.4 |

---

## How It Works

1. **Data Generation** (`preprocess.py`) — `generate_synthetic_dataset()` creates two groups of records with statistically different feature distributions: normal traffic (65%) and suspicious traffic (35%).
2. **Preprocessing** (`preprocess.py`) — Categorical columns (`protocol_type`, `flag`) are label-encoded. All numeric features are standardised using `StandardScaler`.
3. **Training** (`train_model.py`) — An 80/20 stratified train-test split is used. A `RandomForestClassifier` with 100 trees is fitted on the training set.
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score and a Confusion Matrix are computed on the test set and displayed in the dashboard.
5. **Dashboard** (`app.py`) — Streamlit renders charts, metrics and allows CSV upload for inference using the saved model artefacts.

---

## Limitations

- Trained entirely on **synthetic data** — real-world traffic patterns may differ significantly.
- High accuracy reflects the model learning the synthetic data distribution, not real threat intelligence.
- No real-time packet capture or network interface integration.
- Not suitable for production deployment without retraining on validated, real-world data.

---

