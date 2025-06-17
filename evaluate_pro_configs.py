import json
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv


# === CONFIGURACIÓN ===
CSV_PATH = "data/BINANCE-USDT_BRL-100_depth-1749231790356.csv"
BEST_CONFIG = {
    "reward_trade": 0.2208774021033368,
    "reward_hold": -1.157301024543008,
    "reward_profit": 1.5318201509828864,
    "reward_loss": -2.1390810324114664,
    "reward_idle": -0.792946159633988,
    "reward_inventory": 1.7640154739359273,
    "reward_reduce_trade": 2.4433129094748622,
    "reward_hold_hurst_positive": -1.3739119635881387,
    "reward_hold_hurst_negative": 0.07812174131009389,
    "reward_hold_lyap_positive": -1.1087889517735858,
    "reward_hold_lyap_negative": -0.3468591581700773,
    'fee_rate': 0.001  # simulación de Binance
}
N_RUNS = 5
TIMESTEPS = 150_000
results = []

def calculate_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_profit_factor(trades):
    profits = [t[3] for t in trades if t[1] == "SELL" and len(t) > 3 and t[3] > 0]
    losses = [-t[3] for t in trades if t[1] == "SELL" and len(t) > 3 and t[3] < 0]
    total_profit = sum(profits)
    total_loss = sum(losses)
    if total_loss == 0:
        return float('inf') if total_profit > 0 else 0
    return total_profit / total_loss

for run in range(N_RUNS):
    env = SimplifiedTradingEnv(CSV_PATH, reward_config=BEST_CONFIG)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)

    bid = env.df.iloc[env.current_step - 1]["bid"]
    final_equity = env.cash + env.inventory * bid

    results.append({
        "run": run + 1,
        "final_cash": env.cash,
        "final_inventory": env.inventory,
        "final_equity": final_equity,
        "num_trades": len(env.trades),
        "num_sell_trades": len([t for t in env.trades if t[1] == "SELL"])
    })

# Guardar y mostrar resultados
results_df = pd.DataFrame(results)
results_df.to_csv("best_pro_evaluation.csv", index=False)

equity_curve = env.equity_history if hasattr(env, "equity_history") else []
max_drawdown = calculate_max_drawdown(equity_curve) if equity_curve else 0
profit_factor = calculate_profit_factor(env.trades)

print(f"📉 Max Drawdown: {max_drawdown:.2%}")
print(f"📈 Profit Factor: {profit_factor:.2f}")

print("\n✅ Evaluación completada:")
print(results_df)
print("\n📊 Media equity:", results_df["final_equity"].mean())
print("📉 Desvío estándar equity:", results_df["final_equity"].std())
