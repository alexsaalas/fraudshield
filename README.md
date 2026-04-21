# FraudShield

FraudShield es un sistema de detección de fraude en transacciones financieras desarrollado en Python utilizando técnicas de machine learning.

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
│   ├── raw/              # Datos originales
│   └── processed/        # Datos preprocesados
├── models/               # Modelos entrenados (.pkl)
├── src/
│   ├── load_data.py      # Carga de datos
│   ├── eda.py            # Análisis exploratorio
│   ├── preprocess.py     # Limpieza y feature engineering
│   ├── train.py          # Logistic Regression
│   ├── train_random_forest.py
│   └── threshold_analysis.py
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
### 2. Preprocesar datos

```bash
python src/preprocess.py
## 3. Entrenar modelos
python src/train.py
python src/train_random_forest.py

---

## 🔧 Qué se ha corregido

- ✔️ Se han separado los pasos con `###` (mejor lectura)
- ✔️ Cada bloque de comandos va dentro de ```bash```
- ✔️ Se evita texto plano (que en GitHub queda mal)

---

## 🎯 Resultado

En GitHub se verá así:

- Paso 2 → bloque de código limpio  
- Paso 3 → bloque separado  
- Todo ordenado y profesional  

---
## 📌 Validación adicional del modelo

Para reforzar la robustez del proyecto, se añadieron varias capas de validación y análisis:

- **Feature importance** para identificar qué variables influyen más en la detección de fraude
- **Evaluación ampliada** con ROC-AUC, PR-AUC y matriz de confusión
- **Comparación formal de modelos** entre Logistic Regression y Random Forest
- **Cross-validation** para comprobar estabilidad y capacidad de generalización
- **Threshold tuning** para ajustar la sensibilidad del sistema en un problema altamente desbalanceado
- **Ejecución unificada del pipeline** mediante un script global

---

## 🧪 Scripts adicionales

```bash
python src/feature_importance.py
python src/evaluate_model.py
python src/compare_models.py
python src/cv_score.py
python run_all.py