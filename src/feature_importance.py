import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 1. CARGAR MODELO
# =========================
pipeline = joblib.load(BASE_DIR / "models" / "fraudshield_random_forest.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# =========================
# 2. OBTENER NOMBRES DE FEATURES
# =========================
feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("=== TOP 20 FEATURES MÁS IMPORTANTES ===")
print(fi.head(20).to_string(index=False))

# =========================
# 3. GUARDAR RESULTADOS
# =========================
output_path = BASE_DIR / "data" / "processed" / "feature_importance.csv"
fi.to_csv(output_path, index=False)

print(f"\nArchivo guardado en: {output_path}")