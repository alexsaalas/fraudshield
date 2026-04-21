import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR MODELO
# =========================
model_path = BASE_DIR / "models" / "fraudshield_random_forest.pkl"
model = joblib.load(model_path)

print("Modelo cargado correctamente")
print("\nIntroduce los datos de la transacción:\n")

# =========================
# 2. FUNCIONES SEGURAS DE INPUT
# =========================
def input_float(msg):
    while True:
        try:
            return float(input(msg).strip())
        except ValueError:
            print("❌ Introduce un número válido.")

def input_int(msg, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(msg).strip())
            if min_val is not None and val < min_val:
                print(f"❌ Debe ser ≥ {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"❌ Debe ser ≤ {max_val}")
                continue
            return val
        except ValueError:
            print("❌ Introduce un número entero válido.")

def input_str(msg, allowed=None, transform=None):
    while True:
        val = input(msg).strip()
        if transform:
            val = transform(val)
        if val == "":
            print("❌ Este campo no puede estar vacío.")
            continue
        if allowed and val not in allowed:
            print(f"❌ Valores permitidos: {allowed}")
            continue
        return val

# =========================
# 3. PEDIR DATOS AL USUARIO
# =========================
merchant = input_str("Empresa o comercio (ej: Amazon, Zara, Netflix): ")
category = input_str("Categoría (ej: compras, ocio, transporte): ")
amt = input_float("Importe de la transacción (€): ")
gender = input_str("Género (M/F): ", allowed=["M", "F"], transform=lambda x: x.upper())
job = input_str("Profesión: ")
hour = input_int("Hora de la transacción (0-23): ", 0, 23)
day = input_int("Día del mes (1-31): ", 1, 31)
month = input_int("Mes (1-12): ", 1, 12)
age = input_int("Edad: ", 0, 120)

# =========================
# 4. VALORES AUTOMÁTICOS (para demo)
# =========================
state = "MAD"
lat = 40.4168
long = -3.7038
city_pop = 3200000
unix_time = 1700000000
merch_lat = 40.4170
merch_long = -3.7040

# =========================
# 5. CREAR DATAFRAME
# =========================
sample = {
    "merchant": merchant,
    "category": category,
    "amt": amt,
    "gender": gender,
    "state": state,
    "job": job,
    "lat": lat,
    "long": long,
    "city_pop": city_pop,
    "unix_time": unix_time,
    "merch_lat": merch_lat,
    "merch_long": merch_long,
    "hour": hour,
    "day": day,
    "month": month,
    "age": age
}

df_input = pd.DataFrame([sample])

# =========================
# 6. PREDICCIÓN (CON THRESHOLD)
# =========================
probability = model.predict_proba(df_input)[0][1]

# 🔥 AQUÍ ESTÁ LA CLAVE
threshold = 0.3  # puedes ajustar (0.3 recomendado para fraude)
prediction = int(probability >= threshold)

# =========================
# 7. RESULTADO
# =========================
print("\n=== RESULTADO ===")
print(f"Empresa/comercio: {merchant}")
print(f"Categoría: {category}")
print(f"Importe: {amt:.2f} €")
print(f"Probabilidad de fraude: {probability:.4f}")
print(f"Threshold aplicado: {threshold}")

if prediction == 1:
    print("🚨 FRAUDE DETECTADO")
else:
    print("✅ TRANSACCIÓN NORMAL")