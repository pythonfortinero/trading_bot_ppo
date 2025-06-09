import optuna
from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "data/BITSO-XRP_MXN-1000_depth-1748377579235.csv"

# Mejor configuración anterior (Trial 12)
best_reward_config = {
    "reward_trade": 2.0212,
    "reward_hold": -1.9858,
    "reward_profit": 1.7858,
    "reward_loss": -3.5062,
    "reward_idle": -0.3802,
}

# Función para sugerir cerca de un valor
def suggest_near(trial, name, base, delta):
    return trial.suggest_float(name, base - delta, base + delta)

# Objective para Optuna
def objective(trial):
    reward_config = {
        "reward_trade": suggest_near(trial, "reward_trade", best_reward_config["reward_trade"], 0.15),
        "reward_hold": suggest_near(trial, "reward_hold", best_reward_config["reward_hold"], 0.1),
        "reward_profit": suggest_near(trial, "reward_profit", best_reward_config["reward_profit"], 0.25),
        "reward_loss": suggest_near(trial, "reward_loss", best_reward_config["reward_loss"], 0.4),
        "reward_idle": suggest_near(trial, "reward_idle", best_reward_config["reward_idle"], 0.1),
    }

    env = SimplifiedTradingEnv(CSV_PATH, reward_config=reward_config)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50_000)

    # Calculamos el equity final sin necesidad de report()
    env.report()
    final_equity = env.get_final_equity()

    # Guardamos modelo si supera el benchmark anterior
    if final_equity > 1021.45:
        model.save(f"model_equity_{int(final_equity)}.zip")
        with open(f"model_equity_{int(final_equity)}.json", "w") as f:
            import json
            json.dump(reward_config, f, indent=2)

    return final_equity

# Ejecutamos búsqueda más fina
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n Optimización fina completa.")
print("Mejor configuración:")
print(study.best_params)