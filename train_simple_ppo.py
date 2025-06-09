from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env_simple import SimplifiedTradingEnv

# Ruta al CSV de snapshots
csv_path = "data/BITSO-XRP_MXN-1000_depth-1748377579235.csv"

env = SimplifiedTradingEnv(csv_path)
check_env(env)

# Crear y entrenar el modelo
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=150000)

# Guardar modelo entrenado
model.save("ppo_trading_simplificado")

