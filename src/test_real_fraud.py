import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR MODELO
# =========================
model_path = BASE_DIR / "models" / "fraudshield_random_forest.pkl"
model = joblib.load(model_path)

# =========================
# 2. CARGAR DATOS
# =========================
data_path = BASE_DIR / "data" / "processed" / "data_clean.csv"
df = pd.read_csv(data_path)

# =========================
# 3. SELECCIONAR FRAUDE REAL
# =========================
fraud_df = df[df["is_fraud"] == 1]

if fraud_df.empty:
    raise ValueError("❌ No se encontraron casos de fraude en el dataset.")

# Puedes cambiar el índice para probar otros casos
fraud_row = fraud_df.iloc[0]

# Separar X e y
X_test = fraud_row.drop(labels=["is_fraud"]).to_frame().T
y_real = fraud_row["is_fraud"]

# =========================
# 4. PREDICCIÓN CON THRESHOLD
# =========================
probability = model.predict_proba(X_test)[0][1]

# Ajusta aquí el threshold (clave en datasets desbalanceados)
threshold = 0.3
prediction = int(probability >= threshold)

# =========================
# 5. OUTPUT
# =========================
print("=== CASO REAL DEL DATASET ===")
print(X_test.to_string(index=False))

print("\n=== RESULTADO ===")
print(f"Valor real (is_fraud): {y_real}")
print(f"Probabilidad de fraude: {probability:.4f}")
print(f"Threshold aplicado: {threshold}")
print(f"Predicción final: {prediction}")

if prediction == 1:
    print("🚨 FRAUDE DETECTADO")
else:
    print("✅ TRANSACCIÓN NORMAL")