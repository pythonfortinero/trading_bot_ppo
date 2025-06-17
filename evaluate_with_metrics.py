from stable_baselines3 import PPO
from env_simple import SimplifiedTradingEnv

#csv_path = "data/BITSO-XRP_MXN-1000_depth-1748377579235.csv"
csv_path = "data/BITSO-USD_BRL-1000_depth-1748377578952.csv"

model_path = "model_equity_2125"

env = SimplifiedTradingEnv(csv_path, max_steps=500)
model = PPO.load(model_path)

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()

env.report()
#env.plot_metrics()
