# FraudShield

FraudShield es un sistema de detección de fraude en transacciones financieras desarrollado en Python utilizando técnicas de machine learning.

---

## 🎯 Objetivo

Detectar operaciones fraudulentas en un dataset altamente desbalanceado, optimizando el equilibrio entre precisión y recall.

---

## 🧠 Descripción técnica

El proyecto incluye:

- Análisis exploratorio de datos (EDA)
- Preprocesamiento y feature engineering
- Manejo de clases desbalanceadas
- Entrenamiento de modelos de clasificación:
  - Logistic Regression
  - Random Forest
- Evaluación mediante:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
- Optimización del threshold de decisión

---

## 📊 Resultados

### Logistic Regression
- Precision (fraude): 0.035
- Recall (fraude): 0.827
- ROC-AUC: 0.9507

### Random Forest
- Precision (fraude): 0.114
- Recall (fraude): 0.757
- ROC-AUC: 0.9682

👉 El modelo Random Forest mejora significativamente la precisión manteniendo un alto recall.

---

## ⚠️ Problema abordado

Dataset altamente desbalanceado:

- 99.6% transacciones normales
- 0.4% fraude

Se aplicaron técnicas como:

- `class_weight="balanced"`
- ajuste de threshold de decisión

---

## 🏗️ Estructura del proyecto

```bash
fraudshield/
├── data/
│   ├── raw/              # Datos originales (no incluidos)
│   └── processed/        # Datos preprocesados
├── models/               # Modelos entrenados (.pkl)
├── src/
│   ├── load_data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── train_random_forest.py
│   ├── evaluate_model.py
│   ├── feature_importance.py
│   ├── compare_models.py
│   ├── cv_score.py
│   ├── predict.py
│   └── threshold_analysis.py
├── run_all.py
├── README.md
└── requirements.txt
## ⚙️ Tecnologías utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn  
- Joblib  

---

## 🚀 Ejecución del proyecto
### 1. Instalar dependencias
```bash
pip install -r requirements.txt
# 2. Preprocesar datos
python src/preprocess.py
# 3 Entrenamiento de modelos
python src/train.py
python src/train_random_forest.py
# 4. Evaluación completa
python src/evaluate_model.py
# 5. Ejecutar pipeline completo
python run_all.py
# SCRIPTS ADICIONALES
python src/feature_importance.py
python src/compare_models.py
python src/cv_score.py
# RESULTADOS CLAVES (explicados)
El modelo ha sido evaluado utilizando distintas métricas para entender su comportamiento en un problema de fraude altamente desbalanceado.

🔢 Métricas principales
ROC-AUC: 0.9682
PR-AUC: 0.6179
Recall (fraude): ~0.75
Precision (fraude): ~0.11
🧠 ¿Qué significan estas métricas?
🎯 ROC-AUC (0.9682)

Mide la capacidad del modelo para distinguir entre transacciones normales y fraudulentas.

👉 Un valor cercano a 1 indica que el modelo separa muy bien ambos casos.
👉 En este caso, el rendimiento es muy alto.

⚠️ PR-AUC (0.6179)

Es más relevante en datasets desbalanceados.

👉 Mide la calidad al detectar fraude sin generar demasiados errores.
👉 Un valor de 0.61 es muy sólido en este tipo de problema.

🚨 Recall (fraude) ~ 75%

Indica el porcentaje de fraudes detectados.

👉 El modelo detecta aproximadamente 3 de cada 4 fraudes.

✔️ Prioriza seguridad
✔️ Reduce el riesgo de dejar pasar fraude

⚖️ Precision (fraude) ~ 11%

Indica cuántas alertas son realmente fraude.

👉 De cada 100 alertas, ~11 son fraude real.

❗ Es normal debido al fuerte desbalance del dataset

⚠️ Interpretación global

El modelo está optimizado para:

✔️ Detectar la mayor cantidad posible de fraude
❌ A costa de generar falsas alarmas

👉 En fraude real:

Es mejor investigar una transacción normal que permitir un fraude.

🎛️ Ajuste del threshold

El modelo genera probabilidades y se aplica un threshold:

Threshold estándar: 0.5
Threshold utilizado: 0.3

👉 Esto permite:

aumentar recall
detectar más fraude
reducir falsos negativos
🧠 Conclusión

El modelo no se basa en reglas simples (como “importe alto = fraude”),
sino en patrones complejos combinando múltiples variables, como:

comportamiento temporal
tipo de comercio
perfil del cliente

Esto permite una detección más realista y cercana a sistemas antifraude reales.
