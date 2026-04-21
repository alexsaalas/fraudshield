import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR DATOS
# =========================
df = pd.read_csv(BASE_DIR / "data" / "processed" / "data_clean.csv")

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

# =========================
# 2. TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 3. CARGAR MODELO
# =========================
model = joblib.load(BASE_DIR / "models" / "fraudshield_random_forest.pkl")

# =========================
# 4. PREDICCIÓN
# =========================
y_proba = model.predict_proba(X_test)[:, 1]

threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

# =========================
# 5. MÉTRICAS
# =========================
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Threshold usado: {threshold}")