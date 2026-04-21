import pandas as pd
from pathlib import Path

print("Iniciando carga de datos...")

# Ruta absoluta robusta (esto evita TODOS los problemas de rutas)
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "fraudTest.csv"

print(f"Ruta del archivo: {data_path}")

# Cargar dataset
df = pd.read_csv(data_path)

print("Datos cargados correctamente")
print(df.head())
print(df.shape)