import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR DATOS
# =========================
df = pd.read_csv(BASE_DIR / "data" / "processed" / "data_clean.csv")

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

# =========================
# 2. CARGAR MODELO
# =========================
model = joblib.load(BASE_DIR / "models" / "fraudshield_random_forest.pkl")

# =========================
# 3. CROSS VALIDATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("=== CROSS-VALIDATION ROC-AUC ===")
print("Scores por fold:", scores)
print(f"Media: {scores.mean():.4f}")
print(f"Desviación estándar: {scores.std():.4f}")