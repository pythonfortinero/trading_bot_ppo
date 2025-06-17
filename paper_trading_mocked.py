import requests
import time
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import logging
import pandas as pd

CSV_PATH = "data/BINANCE-USDT_BRL-100_depth-1749231790356.csv"

class MockedBinance:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.index = 0

    def get_next_snapshot(self):
        if self.index >= len(self.df):
            return None  # Fin de datos

        row = self.df.iloc[self.index]
        self.index += 1

        bid = float(row['bid'])
        ask = float(row['ask'])
        return {'bid': bid, 'ask': ask, 'spread_percentage': row['spread_percentage']}

class PaperTradingBot:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.inventory = 0.0  # en USDT
        self.cash = 100.0  # saldo inicial en ARS
        self.trades = []
        self.equity_history = []
        self.fee_rate = 0.001  # 0.1%

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        logname = f"paper_trading_{now}.log"

        logging.basicConfig(
            filename=logname,
            filemode='a',
            format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'))

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(console_handler)
        self.logger.info("📈 Iniciando Paper Trading Bot...")

    def _get_observation(self, snapshot):
        bid = snapshot['bid']
        ask = snapshot['ask']
        spread_percentage = snapshot['spread_percentage']
        return np.array([
            bid,
            ask,
            spread_percentage,
            self.inventory,
            self.cash
        ], dtype=np.float32)

    def ejecutar_compra(self, bid, usd):
        ars_needed = usd * bid * (1 + self.fee_rate)
        if self.cash >= ars_needed:
            self.inventory += usd
            self.cash -= ars_needed
            equity = self.calcular_equity(bid)
            self.trades.append(("BUY", usd, bid))
            self.logger.info(f"🟢 BUY: {usd:.2f} USDT @ {bid:.2f} ARS | Cash: {self.cash:.2f}, Inv: {self.inventory:.2f}, Eq: {equity:.2f}")

    def ejecutar_venta(self, ask, usd):
        if self.inventory >= usd:
            self.inventory -= usd
            ars_received = usd * ask * (1 - self.fee_rate)
            self.cash += ars_received

            buy_trades = [t for t in self.trades if t[0] == "BUY"]
            if buy_trades:
                avg_buy_price = sum(t[1] * t[2] for t in buy_trades) / sum(t[1] for t in buy_trades)
                pnl = (ask - avg_buy_price) * usd
            else:
                pnl = 0.0

            equity = self.calcular_equity(ask)
            self.trades.append(("SELL", usd, ask, pnl))
            self.logger.info(f"🔴 SELL: {usd:.2f} USDT @ {ask:.2f} ARS | PnL: {pnl:.2f}, Cash: {self.cash:.2f}, Inv: {self.inventory:.2f}, Eq: {equity:.2f}")

    def calcular_equity(self, bid):
        return self.cash + self.inventory * bid

    def run(self, exchange):
        while True:
            snapshot = exchange.get_next_snapshot()
            if snapshot is None:
                break

            obs = self._get_observation(snapshot)
            action, _ = self.model.predict(obs, deterministic=True)

            if action == 1:
                self.ejecutar_compra(snapshot['bid'], usd=5.0)
            elif action == 2:
                if self.inventory >= 20:
                    self.ejecutar_venta(snapshot['ask'], usd=self.inventory)
                else:
                    self.ejecutar_venta(snapshot['ask'], usd=5.0)

            equity = self.calcular_equity(snapshot['bid'])
            self.equity_history.append(equity)
            self.logger.debug(f"Equity actualizado: {equity:.2f}")

# Ejemplo de uso:
# exchange = MockedBinance(CSV_PATH)
# bot = PaperTradingBot(model_path="ppo_model.zip")
# bot.run(exchange)
