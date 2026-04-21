import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR DATOS
# =========================
df = pd.read_csv(BASE_DIR / "data" / "processed" / "data_clean.csv")

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 2. CARGAR MODELOS
# =========================
models = {
    "Logistic Regression": joblib.load(BASE_DIR / "models" / "fraudshield_logreg.pkl"),
    "Random Forest": joblib.load(BASE_DIR / "models" / "fraudshield_random_forest.pkl")
}

results = []

for model_name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Modelo": model_name,
        "ROC-AUC": round(roc_auc, 4)
    })

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

print("=== COMPARACIÓN DE MODELOS ===")
print(results_df.to_string(index=False))