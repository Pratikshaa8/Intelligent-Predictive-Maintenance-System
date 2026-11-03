# PM-CDSS — Intelligent Predictive-Maintenance Decision Support System

A Streamlit application reconstructed from the uploaded Computational Intelligence Laboratory Tasks 1–10 and the AI4I 2020 dataset.

## Features
- Individual machine assessment using sensor inputs
- Priority-based expert rules R1–R6 and a readable rule trace
- XGBoost primary model with class-imbalance handling
- Three-hidden-layer neural-network approximation (32, 16, 8)
- Risk fusion: 0.5 × ML + 0.3 × DL + 0.2 × expert score
- Explainability using local feature perturbation impacts and what-if guidance
- Fleet CSV scoring, risk analytics and downloadable scored CSV
- PDF maintenance report and JSON prediction export
- Responsible AI notes and benchmark metrics from Task 10

## Run on Windows
1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/
2. Extract this ZIP.
3. Open Command Prompt in the extracted `PM_CDSS_Project` folder.
4. Create and activate a virtual environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

5. Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the app:

```bat
streamlit run app.py
```

7. Open the local URL shown in the terminal (normally http://localhost:8501).

## Run on macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Optional commands
Train models and print hold-out ROC-AUC:
```bash
python train_models.py
```

Run rule and validation tests:
```bash
python -m pytest -q
```

## Input CSV format
Fleet upload expects these columns:
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

`UDI` is optional. `Machine failure` is optional and, if present, is used only as the observed-failure evidence input for R5 in this prototype.

## Important implementation notes
- The original reports refer to `best_model.pkl`, `dnn_model.h5` and `dnn_scaler.pkl`, but these trained artifacts were not included in the uploads. The app therefore trains replacement models from the included `ai4i2020.csv` on first launch and caches them for the session.
- The original Task 7 DNN used Keras with Dense(32, 16, 8), BatchNorm, Dropout and L2. This runnable reconstruction uses scikit-learn MLPClassifier with hidden layers (32, 16, 8); it is a practical approximation, not the original trained Keras network.
- `Power_Deviation_W` is implemented as mechanical power minus 6000 W. The original Task 5 formula was not recoverable from the supplied PDF text, so this assumption is explicit and easy to change in `src/pipeline.py`.
- Local feature impacts are perturbation-based approximations, not exact SHAP values. Task 8/10 global SHAP and trust figures are displayed as reported reference findings.
- Reported benchmark metrics are shown separately from newly trained model performance.
- The system is decision support only. A maintenance engineer must approve actions.

## Project structure
```text
PM_CDSS_Project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
├── Dockerfile
├── data/
│   └── ai4i2020.csv
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   └── report.py
└── tests/
    └── test_pipeline.py
```
