import json
import os
import pandas as pd
from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv
import numpy as np

# Rutas
CSV_PATH = "data/BITSO-XRP_MXN-1000_depth-1748377579235.csv"
CONFIG_DIR = "models"  # Carpeta con config_XXXX.json
N_RUNS = 3  # Cuántas veces entrenar cada config

results = []

# Buscar archivos JSON de configuración
config_files = [f for f in os.listdir(CONFIG_DIR) if f.startswith("config_") and f.endswith(".json")]
config_files.sort()  # Opcional: orden alfabético

for cfg_file in config_files:
    with open(os.path.join(CONFIG_DIR, cfg_file)) as f:
        reward_config = json.load(f)

    print(f"📊 Evaluando configuración: {cfg_file}")
    equities = []

    for i in range(N_RUNS):
        env = SimplifiedTradingEnv(CSV_PATH, reward_config=reward_config)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50_000)

        bid = env.df.iloc[env.current_step - 1]["bid"]
        equity = env.cash + env.inventory * bid
        equities.append(equity)

    mean_eq = np.mean(equities)
    std_eq = np.std(equities)

    results.append({
        "config_file": cfg_file,
        "mean_equity": mean_eq,
        "std_equity": std_eq,
        **reward_config  # Despliega cada parámetro como columna
    })

# Guardar a CSV
df = pd.DataFrame(results)
df.to_csv("best_config_evaluation.csv", index=False)
print("\nEvaluación completa. Resultados guardados en best_config_evaluation.csv")
