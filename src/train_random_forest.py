import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR DATOS
# =========================
data_path = BASE_DIR / "data" / "processed" / "data_clean.csv"
df = pd.read_csv(data_path)

print("Datos cargados:", df.shape)

# =========================
# 2. SEPARAR X / y
# =========================
y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

# =========================
# 3. TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# =========================
# 4. COLUMNAS
# =========================
categorical_cols = ["merchant", "category", "gender", "job", "state"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# =========================
# 5. PREPROCESADO
# =========================
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

# =========================
# 6. MODELO RANDOM FOREST
# =========================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

# =========================
# 7. ENTRENAR
# =========================
print("Entrenando Random Forest...")
pipeline.fit(X_train, y_train)

# =========================
# 8. EVALUAR
# =========================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

# =========================
# 9. GUARDAR MODELO
# =========================
model_path = BASE_DIR / "models" / "fraudshield_random_forest.pkl"
joblib.dump(pipeline, model_path)

print(f"Modelo guardado en: {model_path}")