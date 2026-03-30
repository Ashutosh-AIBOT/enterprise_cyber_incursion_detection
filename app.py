import os

from dashboard_core import ProjectConfig, run_project_app


CONFIG = ProjectConfig(
    key="Project 11",
    title="Enterprise Cyber-Incursion Detection",
    subtitle="Hybrid anomaly and imbalanced-threat intelligence for NSL-KDD",
    icon="🛡️",
    domain="Enterprise security operations",
    objective=(
        "Detect known and unknown cyber intrusions using unsupervised anomaly detectors "
        "plus imbalanced supervised models."
    ),
    business_value=(
        "Reduces false negatives in high-stakes threat hunting and improves SOC response speed "
        "with explainable packet-level risk signals."
    ),
    prediction_label="Intrusion Risk",
    highlights=[
        "Hybrid detection stack: One-Class SVM, LOF, and deep autoencoder anomaly engines.",
        "Imbalance-aware supervised learning with ADASYN, Tomek Links, and BorderlineSMOTE.",
        "Unknown-threat handling designed for needle-in-a-haystack cybersecurity events.",
        "Explainability module with LIME and permutation importance for packet-level decisions.",
        "Deployment-ready modular architecture for SOC and SecOps workflows.",
    ],
)


run_project_app(CONFIG, os.path.dirname(__file__))
