import optuna
from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv
import pandas as pd
import warnings
import json
import os

warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "data/BITSO-XRP_MXN-1000_depth-1748377579235.csv"
SAVE_DIR = "models_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# Configuración ganadora
best_config = {
    "reward_trade": 0.5147365863109863,
    "reward_hold": -1.6694517294809623,
    "reward_profit": 0.718663344427364,
    "reward_loss": -1.4570597919158168,
    "reward_idle": -0.2622686091836593,
}

# Margen de ±5%
def get_margin(value, pct=0.05):
    return max(abs(value) * pct, 0.01)

# Sugerir dentro de ±5% del valor base
def suggest_near(trial, name, base):
    delta = get_margin(base)
    return trial.suggest_float(name, base - delta, base + delta)

# Objective con entrenamiento largo
def objective(trial):
    reward_config = {
        k: suggest_near(trial, k, best_config[k])
        for k in best_config
    }

    env = SimplifiedTradingEnv(CSV_PATH, reward_config=reward_config)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=150_000)

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

    # Guardar si cumple criterios
    equity_str = f"{int(final_equity)}_{trial.number}"
    model.save(f"{SAVE_DIR}/model_{equity_str}.zip")
    with open(f"{SAVE_DIR}/config_{equity_str}.json", "w") as f:
        json.dump(reward_config, f, indent=2)

    return final_equity

# Estudio Optuna
study = optuna.create_study(
    direction="maximize",
    study_name="final_opt",
    storage="sqlite:///final_optuna_study.db",
    load_if_exists=True
)
study.optimize(objective, n_trials=30)

# Guardar resultados
df_trials = study.trials_dataframe()
df_trials.to_csv("optuna_final_results.csv", index=False)

print("\n Optimización finalizada.")
print(" Mejor configuración:")
print(study.best_params)
