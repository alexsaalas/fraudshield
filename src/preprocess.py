import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "fraudTest.csv"

df = pd.read_csv(data_path)

print("Iniciando preprocesamiento...")

# =========================
# 1. ELIMINAR COLUMNAS INÚTILES
# =========================
cols_to_drop = [
    "Unnamed: 0",
    "trans_num",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "zip"
]

df = df.drop(columns=cols_to_drop)

# =========================
# 2. FECHAS
# =========================
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Crear nuevas features
df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["month"] = df["trans_date_trans_time"].dt.month

# Edad del usuario
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

# Eliminar columnas originales de fecha
df = df.drop(columns=["trans_date_trans_time", "dob"])

# =========================
# 3. TARGET
# =========================
y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

print("Preprocesamiento completado")
print(f"X shape: {X.shape}")
print(f"y distribución:\n{y.value_counts()}")

# Guardar datos procesados
output_path = BASE_DIR / "data" / "processed" / "data_clean.csv"
df.to_csv(output_path, index=False)

print(f"Datos guardados en: {output_path}")