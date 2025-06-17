import optuna
from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv

CSV_PATH = "data/BINANCE-USDT_BRL-100_depth-1749231790356.csv"

def objective(trial):
    reward_config = {
        "reward_trade": trial.suggest_float("reward_trade", 0.1, 0.3),
        "reward_hold": trial.suggest_float("reward_hold", -2.0, 0.0),
        "reward_profit": trial.suggest_float("reward_profit", 0.5, 5.0),
        "reward_loss": trial.suggest_float("reward_loss", -5.0, -0.1),
        "reward_idle": trial.suggest_float("reward_idle", -1.0, 0.0),
        "reward_inventory": trial.suggest_float("reward_inventory", 1.0, 2.0),
        "reward_reduce_trade": trial.suggest_float("reward_reduce_trade", 1.0, 5.0),
        "reward_hold_hurst_positive": trial.suggest_float("reward_hold_hurst_positive", -2.0, 2.0),
        "reward_hold_hurst_negative": trial.suggest_float("reward_hold_hurst_negative", -2.0, 2.0),
        "reward_hold_lyap_positive": trial.suggest_float("reward_hold_lyap_positive", -2.0, 2.0),
        "reward_hold_lyap_negative": trial.suggest_float("reward_hold_lyap_negative", -2.0, 2.0),
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

    # 🛑 Filtrado por condiciones
    if final_cash < 900 or final_equity <= 1000:
        return -1.0  # Penalización fuerte para que Optuna descarte este trial

    # 💾 Guardar buen modelo
    model.save(f"models/model_equity_{int(final_equity)}.zip")
    with open(f"models/model_equity_{int(final_equity)}.json", "w") as f:
        import json
        json.dump(reward_config, f, indent=2)

    return final_equity 

# Ejecutar optimización
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Mostrar mejor configuración
print("Mejor configuración encontrada:")
print(study.best_params)