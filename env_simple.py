# env_simple.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hurst import compute_Hc
import nolds

class SimplifiedTradingEnv(gym.Env):
    def __init__(self, csv_path, reward_config=None, max_steps=500):
        super().__init__()

        self.reward_config = reward_config or {
            "reward_trade": 1.0,
            "reward_hold": -1.0,
            "reward_profit": 1.0,
            "reward_loss": -2.0,
            "reward_idle": -0.1,
        }
        self.df = pd.read_csv(csv_path)
        self.max_steps = max_steps

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.start_step = 0
        self.cash = 1000.0
        self.inventory = 0
        self.inventory_value = 0.0
        self.last_equity = 1000.0
        self.total_reward = 0
        self.trades = []

    def reset(self, seed=None, options=None):
        self.start_step = np.random.randint(0, len(self.df) - self.max_steps)
        self.current_step = self.start_step
        self.cash = 1000.0
        self.inventory = 0
        self.inventory_value = 0.0
        self.last_equity = 1000.0
        self.total_reward = 0
        self.trades = []
        self.cash_history = []
        self.inventory_history = []
        self.reward_history = []
        self.prev_inventory = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row['bid'],
            row['ask'],
            row['spread_percentage'],
            self.inventory,
            self.cash
        ], dtype=np.float32)

    def _compute_hurst_lyap(self):
        start = max(0, self.current_step - 50)
        end = self.current_step
        window = self.df['bid'].iloc[start:end].values

        if len(window) < 20:
            return 0.5, 0.1  # valores neutros

        try:
            H, _, _ = compute_Hc(window, kind='price')
        except:
            H = 0.5

        try:
            lyap = nolds.lyap_r(window)
        except:
            lyap = 0.1

        return H, lyap

    def step(self, action):
        row = self.df.iloc[self.current_step]
        bid = float(row['bid'])
        ask = float(row['ask'])

        done = False
        reward = 0
        hurst, lyap = self._compute_hurst_lyap()

        # Compra si hay cash
        if action == 1 and self.cash >= ask:
            self.inventory += 1
            self.cash -= ask
            self.inventory_value += ask
            reward += self.reward_config["reward_trade"]  # premio por comprar
            reward -= 0.06 * self.inventory
            self.trades.append((self.current_step, "BUY", ask))

        elif action == 2 and self.inventory > 0:
            avg_price = self.inventory_value / (self.inventory + 1e-8)
            self.inventory_value -= avg_price
            diff = bid - avg_price

            if diff > 0:
                base_gain = bid * self.inventory
                reward += self.reward_config["reward_profit"] * base_gain
                if hurst < 0.45:
                    reward += 1.0  # bonus específico
                self.trades.append((self.current_step, "SELL", bid, base_gain))
                self.cash += bid * self.inventory
                self.inventory = 0
            else:
                reward += self.reward_config["reward_loss"] * abs(diff)

            if (self.cash + self.inventory * bid) > 1000:
                reward += 100

        elif action == 0:
            reward += self.reward_config["reward_idle"]
            if hurst > 0.55:
                reward += self.reward_config["reward_hold"]
            if lyap < 0.2:
                reward += self.reward_config["reward_hold"]
            if hurst < 0.45:
                reward += self.reward_config["reward_hold"]
            if lyap > 0.5:
                reward += self.reward_config["reward_hold"]

        # Premio por reducir inventario
        if self.inventory < self.prev_inventory:
            reward += self.reward_config["reward_trade"] * 0.1 * (self.prev_inventory - self.inventory)
        self.prev_inventory = self.inventory

        # Recompensa por equity
        equity = self.cash + self.inventory * bid
        reward += (equity - self.last_equity) * 0.1
        self.last_equity = equity

        # Volatilidad local para escalar
        window = self.df['bid'].iloc[max(0, self.current_step - 10):self.current_step]
        if len(window) > 1:
            volatility = np.std(window)
            reward *= 1 + (volatility * 2)

        # Exploración
        reward += np.random.normal(0, 0.01)

        # Finalizar episodio
        self.current_step += 1
        if self.current_step - self.start_step >= self.max_steps:
            if (self.cash + self.inventory * bid) > 1000 and len(self.trades) > 0:
                reward += 1000
            done = True

        return self._get_observation(), reward, done, False, {}


    def render(self):
        print(f"Step: {self.current_step} | Cash: {self.cash:.2f} | Inventory: {self.inventory} | Total Reward: {self.total_reward:.4f}")

    def report(self):
        print("\n--- Trade History ---")
        for trade in self.trades:
            print(trade)
        print(f"\nFinal Cash: {self.cash:.2f}")
        print(f"Final Inventory: {self.inventory}")
        self.final_equity = self.cash + self.inventory * self.df.iloc[self.current_step]['bid']
        print(f"Final Equity: {self.final_equity:.2f}")
    
    def get_final_equity(self):
        return self.final_equity  # o la lógica que uses para calcularla

    def plot_metrics(self):
        steps = list(range(len(self.cash_history)))

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(steps, self.cash_history, label="Cash")
        axs[0].set_ylabel("Cash")
        axs[0].legend()

        axs[1].plot(steps, self.inventory_history, label="Inventory", color="orange")
        axs[1].set_ylabel("Inventory")
        axs[1].legend()

        axs[2].plot(steps, self.reward_history, label="Cumulative PnL", color="green")
        axs[2].set_ylabel("PnL")
        axs[2].set_xlabel("Step")
        axs[2].legend()

        plt.tight_layout()
        plt.show()
