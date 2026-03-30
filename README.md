# Enterprise Cyber-Incursion Detection (Project 11)

End-to-end Security Intelligence System focused on extreme class imbalance and unknown-threat detection using NSL-KDD.

## What Is Implemented
- Hybrid anomaly detection:
  - One-Class SVM
  - Local Outlier Factor (LOF)
  - Deep AutoEncoder (PyTorch)
- Imbalanced supervised learning:
  - ADASYN
  - Tomek Links
  - BorderlineSMOTE
- Modular balancing orchestrator: `src/sampling_manager.py`
- Explainability module: `src/explainability.py`
  - LIME local explanations
  - Permutation importance
- Streamlit enterprise dashboard: `app.py`

## Folder Structure
- `src/data_loader.py`: NSL-KDD load and fallback synthetic data
- `src/preprocessing.py`: train/test split and encoding/scaling pipeline
- `src/sampling_manager.py`: imbalance handling strategies
- `src/anomaly_models.py`: unsupervised anomaly suite
- `src/supervised_models.py`: supervised model suite
- `src/explainability.py`: LIME + permutation importance
- `src/train_pipeline.py`: full artifact generation flow

## Dataset Placement
Put NSL-KDD files in:
- `data/raw/KDDTrain+.txt`
- `data/raw/KDDTest+.txt`

If missing, synthetic fallback data is used to keep pipeline runnable.

## Run (Conda `ml-env`)
```bash
conda activate ml-env
cd 11_enterprise_cyber_incursion_detection
pip install -r requirements.txt
python -m src.train_pipeline
streamlit run app.py
```

## Deployment
`render.yaml` is included for direct Render deployment.

# enterprise_cyber_incursion_detection
