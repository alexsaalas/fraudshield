import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "fraudTest.csv"

df = pd.read_csv(data_path)

print("=== INFORMACIÓN GENERAL ===")
print(f"Dimensiones: {df.shape}")
print("\nColumnas:")
print(df.columns.tolist())

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== PRIMERAS FILAS ===")
print(df.head())

print("\n=== VALORES NULOS ===")
print(df.isnull().sum().sort_values(ascending=False))

print("\n=== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO ===")
print(df["is_fraud"].value_counts())

print("\n=== DISTRIBUCIÓN NORMALIZADA (%) ===")
print((df["is_fraud"].value_counts(normalize=True) * 100).round(4))

print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe(include="all").transpose())