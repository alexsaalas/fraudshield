import subprocess

commands = [
    ["python", "src/preprocess.py"],
    ["python", "src/train.py"],
    ["python", "src/train_random_forest.py"],
    ["python", "src/evaluate_model.py"],
    ["python", "src/feature_importance.py"],
    ["python", "src/compare_models.py"],
    ["python", "src/cv_score.py"]
]

for command in commands:
    print(f"\nEjecutando: {' '.join(command)}")
    subprocess.run(command, check=True)

print("\n✅ Pipeline completo ejecutado correctamente.")