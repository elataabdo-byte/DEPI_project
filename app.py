import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Sentinel | Predictive Monitoring",
    page_icon="🛡️",
    layout="wide"
)

st.title("Sentinel")
st.subheader("AI-Powered Predictive Monitoring Platform")

st.divider()

st.info(
    "Welcome to Sentinel. "
    "This is the initial public deployment of the platform."
)

col1, col2, col3 = st.columns(3)

col1.metric("Platform", "Online")
col2.metric("Deployment", "Public Preview")
col3.metric("Pipeline", "Not Connected")

st.divider()

st.header("About Sentinel")

st.write(
    """
    Sentinel is an AI-powered predictive monitoring platform
    designed to analyze industrial sensor data, detect anomalies,
    and support predictive maintenance decisions.
    """
)

st.header("Platform Modules")

st.markdown("""
- **Stage 1:** Data preprocessing and windowing.
- **Stage 2:** Anomaly detection.
- **Stage 3:** Diagnosis and root cause analysis.
- **Streaming:** Historical data replay.
- **Alerts:** Email notifications for critical conditions.
""")

st.warning(
    "This is a temporary deployment preview. "
    "The analysis pipeline and email alerts are not connected yet."
)

st.caption(
    f"Sentinel | Initial Deployment | {datetime.now().strftime('%Y-%m-%d')}"
)
