import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar datos
df = pd.read_csv(BASE_DIR / "data/processed/data_clean.csv")

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Cargar modelo
model = joblib.load(BASE_DIR / "models/fraudshield_logreg.pkl")

# Probabilidades
y_proba = model.predict_proba(X_test)[:, 1]

print("Threshold | Precision | Recall | F1")

# Probar distintos thresholds
for threshold in np.arange(0.1, 0.9, 0.1):
    y_pred = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{threshold:.2f}      | {precision:.4f}    | {recall:.4f} | {f1:.4f}")