"""
app.py
------
CyberShield AI – AI-Based Threat Detection System
Streamlit dashboard for classifying network traffic as Normal or Suspicious.

How to run:
    streamlit run app.py

DISCLAIMER
----------
This project uses SYNTHETIC (artificially generated) data. It is built for
educational / demonstration purposes only. It does NOT perform real-time
network monitoring and cannot guarantee detection of actual cyberattacks.
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from preprocess import FEATURE_COLUMNS, generate_synthetic_dataset, preprocess_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/encoders.pkl"
DATA_PATH    = "data/network_traffic.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load trained artefacts. Trains automatically if not found."""
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found – training now. This takes a few seconds …")
        from train_model import train
        train()
    clf      = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return clf, scaler, encoders


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset():
    if not os.path.exists(DATA_PATH):
        generate_synthetic_dataset(save_path=DATA_PATH)
    return pd.read_csv(DATA_PATH)


def predict_dataframe(df: pd.DataFrame, clf, scaler, encoders):
    """
    Apply saved encoders/scaler to df and return predictions.
    Missing or unknown categories are handled gracefully.
    """
    df = df.copy()

    # Encode with saved encoders (unknown values → most-frequent class)
    for col in ["protocol_type", "flag"]:
        le: LabelEncoder = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
        df[col] = le.transform(df[col].astype(str))

    X = df[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)
    proba = clf.predict_proba(X_scaled)[:, 1]   # P(suspicious)
    return preds, proba


def metric_card(label: str, value, colour: str = "#3b82d4"):
    """Render a simple metric tile using HTML."""
    st.markdown(
        f"""
        <div style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;
                    padding:16px 20px;text-align:center;">
          <div style="font-size:13px;color:#57606a;margin-bottom:4px;">{label}</div>
          <div style="font-size:28px;font-weight:700;color:{colour};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.image(
        "https://img.shields.io/badge/CyberShield_AI-Threat_Detection-blue?style=for-the-badge&logo=shield",
        use_column_width=True,
    )
    st.sidebar.title("🛡️ CyberShield AI")
    st.sidebar.caption("AI-Based Threat Detection System")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Analyze Traffic", "📊 Model Evaluation", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>⚠️ **Disclaimer:** This system uses synthetic data for "
        "educational purposes only. It does not perform real-time monitoring "
        "or guarantee detection of actual cyberattacks.</small>",
        unsafe_allow_html=True,
    )
    return page


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_dashboard(clf, scaler, encoders):
    st.title("🛡️ CyberShield AI — Dashboard")
    st.markdown("Real-time-style overview of the **synthetic** training dataset.")

    df = load_dataset()
    preds, proba = predict_dataframe(df, clf, scaler, encoders)
    df["prediction"]  = preds
    df["threat_score"] = proba

    label_map = {0: "Normal", 1: "Suspicious"}
    df["status"] = df["prediction"].map(label_map)

    # Summary metrics
    total      = len(df)
    suspicious = int((df["prediction"] == 1).sum())
    normal     = total - suspicious
    avg_score  = f"{df['threat_score'].mean():.2%}"

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Records",     total,      "#3b82d4")
    with c2: metric_card("Normal",            normal,     "#22c55e")
    with c3: metric_card("Suspicious",        suspicious, "#ef4444")
    with c4: metric_card("Avg Threat Score",  avg_score,  "#f59e0b")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Traffic Distribution")
        fig_pie = px.pie(
            df, names="status",
            color="status",
            color_discrete_map={"Normal": "#22c55e", "Suspicious": "#ef4444"},
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Threat Score Distribution")
        fig_hist = px.histogram(
            df, x="threat_score", color="status",
            color_discrete_map={"Normal": "#22c55e", "Suspicious": "#ef4444"},
            nbins=40, barmode="overlay", opacity=0.7,
            labels={"threat_score": "Threat Score (probability)"},
        )
        fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Protocol Type Breakdown")
    proto_counts = df.groupby(["protocol_type", "status"]).size().reset_index(name="count")
    fig_bar = px.bar(
        proto_counts, x="protocol_type", y="count", color="status", barmode="group",
        color_discrete_map={"Normal": "#22c55e", "Suspicious": "#ef4444"},
        labels={"protocol_type": "Protocol", "count": "Record Count"},
    )
    fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Recent suspicious records
    st.subheader("🚨 Recent Suspicious Records")
    suspicious_df = df[df["prediction"] == 1].sort_values("threat_score", ascending=False).head(10)
    if suspicious_df.empty:
        st.success("No suspicious records detected.")
    else:
        st.dataframe(
            suspicious_df[FEATURE_COLUMNS + ["threat_score", "status"]].reset_index(drop=True),
            use_container_width=True,
        )


def page_analyze(clf, scaler, encoders):
    st.title("🔍 Analyze Network Traffic")
    st.markdown(
        "Upload a CSV file containing network traffic records to classify them "
        "as **Normal** or **Suspicious**."
    )

    # Expected columns notice
    with st.expander("📋 Expected CSV columns", expanded=False):
        st.markdown(
            "Your CSV must include the following columns:\n\n"
            + "\n".join(f"- `{c}`" for c in FEATURE_COLUMNS)
            + "\n\n**protocol_type** values: `tcp`, `udp`, `icmp`  \n"
            "**flag** values: `SF`, `REJ`, `S0`, `RSTO`"
        )
        # Provide a sample download
        sample = pd.DataFrame([{
            "duration": 2.5, "src_bytes": 500, "dst_bytes": 1200,
            "packet_count": 15, "protocol_type": "tcp", "flag": "SF",
            "same_srv_rate": 0.85, "diff_srv_rate": 0.05, "srv_count": 8,
        }])
        st.download_button(
            "⬇️ Download sample CSV",
            data=sample.to_csv(index=False),
            file_name="sample_traffic.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # Validate columns
        missing = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        preds, proba = predict_dataframe(user_df, clf, scaler, encoders)
        result_df = user_df[FEATURE_COLUMNS].copy().reset_index(drop=True)
        result_df["threat_score"] = proba.round(4)
        result_df["prediction"]   = ["Suspicious" if p == 1 else "Normal" for p in preds]

        total      = len(result_df)
        n_sus      = int((result_df["prediction"] == "Suspicious").sum())
        n_norm     = total - n_sus

        st.markdown("---")
        st.subheader("Analysis Results")

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Total Records", total)
        with c2: metric_card("Normal", n_norm, "#22c55e")
        with c3: metric_card("Suspicious", n_sus, "#ef4444")

        st.markdown("---")

        # Threat alerts
        if n_sus > 0:
            st.error(f"🚨 **{n_sus} suspicious record(s) detected!** Review the table below.")
        else:
            st.success("✅ All records appear to be normal traffic.")

        # Results table — plain st.dataframe avoids the pandas Styler / Arrow
        # serialisation incompatibility that caused blank rows in Streamlit 1.41+
        # with pandas 2.2 object-dtype columns.  Column config provides the
        # visual cue that Styler row-colouring previously attempted.
        st.dataframe(
            result_df,
            use_container_width=True,
            column_config={
                "prediction": st.column_config.TextColumn(
                    "Prediction",
                    help="Normal or Suspicious",
                ),
                "threat_score": st.column_config.NumberColumn(
                    "Threat Score",
                    format="%.4f",
                    help="Probability of suspicious traffic (0–1)",
                ),
            },
        )

        # Download results
        st.download_button(
            "⬇️ Download results CSV",
            data=result_df.to_csv(index=False),
            file_name="cybershield_results.csv",
            mime="text/csv",
        )

        # Charts
        col_l, col_r = st.columns(2)
        with col_l:
            fig_pie = px.pie(
                result_df, names="prediction",
                color="prediction",
                color_discrete_map={"Normal": "#22c55e", "Suspicious": "#ef4444"},
                title="Traffic Classification",
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_r:
            fig_scatter = px.scatter(
                result_df.reset_index(), x="index", y="threat_score",
                color="prediction",
                color_discrete_map={"Normal": "#22c55e", "Suspicious": "#ef4444"},
                title="Threat Score per Record",
                labels={"index": "Record #", "threat_score": "Threat Score"},
            )
            fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b",
                                   annotation_text="Decision Threshold (0.5)")
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("👆 Upload a CSV file to begin analysis.")


def page_evaluation(clf, scaler, encoders):
    st.title("📊 Model Evaluation")
    st.markdown(
        "Performance of the **Random Forest** classifier on the held-out 20% test split "
        "of the synthetic dataset."
    )

    # Re-evaluate on the fly so metrics are always current
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score,
        precision_score, recall_score,
    )
    from sklearn.model_selection import train_test_split

    df = load_dataset()
    X, y, _, _ = preprocess_features(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = clf.predict(X_test)
    proba  = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Accuracy",  f"{acc:.2%}")
    with c2: metric_card("Precision", f"{prec:.2%}", "#7c5cd8")
    with c3: metric_card("Recall",    f"{rec:.2%}",  "#f59e0b")
    with c4: metric_card("F1-Score",  f"{f1:.2%}",   "#22c55e")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Confusion Matrix")
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Suspicious"],
            y=["Actual Normal",    "Actual Suspicious"],
            colorscale="Blues",
            showscale=True,
        )
        fig_cm.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r:
        st.subheader("Feature Importances")
        importances = clf.feature_importances_
        feat_df = pd.DataFrame({
            "Feature":    FEATURE_COLUMNS,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)
        fig_feat = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
        )
        fig_feat.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    st.subheader("Full Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Normal", "Suspicious"])
    st.code(report, language="text")


def page_about():
    st.title("ℹ️ About CyberShield AI")

    st.markdown(
        """
        ## What is CyberShield AI?
        **CyberShield AI** is an educational cybersecurity demonstration project built
        during an IBM internship. It uses **Machine Learning** to classify network traffic
        records as *Normal* or *Suspicious* based on behavioural features.

        ---
        ## How does it work?
        1. **Synthetic dataset** – A labelled dataset is generated with realistic-looking
           network features (duration, byte counts, packet counts, protocol type, etc.).
        2. **Preprocessing** – Categorical features are label-encoded; numeric features
           are standardised with `StandardScaler`.
        3. **Random Forest classifier** – An ensemble of 100 decision trees is trained on
           80% of the data and evaluated on the remaining 20%.
        4. **Dashboard** – The Streamlit app visualises predictions, threat scores,
           protocol breakdowns and model evaluation metrics.

        ---
        ## Technology Stack
        | Component | Library |
        |-----------|---------|
        | Dashboard | Streamlit |
        | ML Model  | Scikit-learn (Random Forest) |
        | Data      | Pandas / NumPy |
        | Charts    | Plotly |
        | Persistence | Joblib |

        ---
        ## ⚠️ Limitations & Disclaimer
        > **This project uses completely synthetic (artificially generated) data.**

        - The dataset does **not** represent real network traffic from any organisation.
        - The model is trained and tested on the *same distribution* of synthetic data,
          so high accuracy metrics reflect pattern recognition within that distribution —
          not real-world generalisation.
        - This system does **not** perform real-time network monitoring.
        - It **cannot** guarantee the detection of actual cyberattacks.
        - It should **not** be deployed in a production security environment without
          retraining on labelled, real-world traffic data and extensive validation.

        This project is intended solely for **learning and demonstration** purposes.

        ---
        ## Project Structure
        ```
        CyberShield_AI/
        ├── app.py            ← Streamlit dashboard (this file)
        ├── train_model.py    ← Model training & evaluation
        ├── preprocess.py     ← Dataset generation & feature preprocessing
        ├── requirements.txt  ← Python dependencies
        ├── README.md         ← Setup & usage instructions
        ├── data/
        │   └── network_traffic.csv   ← Synthetic dataset
        └── models/
            ├── rf_model.pkl          ← Trained Random Forest
            ├── scaler.pkl            ← Fitted StandardScaler
            └── encoders.pkl          ← Fitted LabelEncoders
        ```

        ---
        *Built with ❤️ for IBM Internship · CyberShield AI v1.0*
        """
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    clf, scaler, encoders = load_model()
    page = sidebar()

    if page == "🏠 Dashboard":
        page_dashboard(clf, scaler, encoders)
    elif page == "🔍 Analyze Traffic":
        page_analyze(clf, scaler, encoders)
    elif page == "📊 Model Evaluation":
        page_evaluation(clf, scaler, encoders)
    else:
        page_about()


if __name__ == "__main__":
    main()
