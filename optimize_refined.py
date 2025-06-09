import optuna
from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv
import pandas as pd
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "data/BITSO-XRP_MXN-1000_depth-1748377579235.csv"

# Mejor configuración hallada en trial 42
best_config = {
    "reward_trade": 0.5147,
    "reward_hold": -1.6695,
    "reward_profit": 0.7187,
    "reward_loss": -1.4571,
    "reward_idle": -0.2623
}

# Margen para explorar alrededor de cada parámetro
margins = {
    "reward_trade": 0.1,
    "reward_hold": 0.2,
    "reward_profit": 0.15,
    "reward_loss": 0.2,
    "reward_idle": 0.1
}

# Función para sugerir valores cerca de la configuración ganadora
def suggest_near(trial, name, base, delta):
    return trial.suggest_float(name, base - delta, base + delta)

# Función objetivo
def objective(trial):
    reward_config = {
        k: suggest_near(trial, k, best_config[k], margins[k])
        for k in best_config
    }

    env = SimplifiedTradingEnv(CSV_PATH, reward_config=reward_config)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50_000)

    env.report()
    final_cash = env.cash
    final_inventory = env.inventory
    final_equity = env.get_final_equity()

    # 🔎 Contar trades SELL
    sell_trades = [t for t in env.trades if t[1] == "SELL"]
    num_sells = len(sell_trades)

    # Filtros de calidad
    if final_cash < 900 or final_inventory >= 2 or final_equity <= 1000 or num_sells < 5:
        return -1.0

    # Guardar modelo y config si pasó los filtros
    equity_str = f"{int(final_equity)}_{trial.number}"
    model.save(f"models/model_equity_{equity_str}.zip")
    with open(f"models/config_{equity_str}.json", "w") as f:
        json.dump(reward_config, f, indent=2)

    return final_equity

# Crear y optimizar
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Guardar resultados en CSV
df_trials = study.trials_dataframe()
df_trials.to_csv("optuna_refined_results.csv", index=False)

print("\n🏁 Optimización refinada completa.")
print("Mejor configuración:")
print(study.best_params)
