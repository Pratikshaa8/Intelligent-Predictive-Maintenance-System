from pathlib import Path
from src.pipeline import load_system
ROOT=Path(__file__).parent
system=load_system(ROOT/"data"/"ai4i2020.csv")
print(system["model_note"])
print(f"Rows: {system['dataset_rows']:,}; failures: {system['failure_count']:,}; holdout XGBoost ROC-AUC: {system['holdout_auc']:.4f}")
