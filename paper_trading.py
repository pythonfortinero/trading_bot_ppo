import requests
import time
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import logging

class PaperTradingBot:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.inventory = 0.0  # en USDT
        self.cash = 5000.0  # saldo inicial en ARS
        self.trades = []
        self.equity_history = []

        self.fee_rate = 0.001  # Fee de Binance (0.1%)

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
        self.logger.info("Iniciando Paper Trading Bot...")

    def get_orderbook_snapshot(self):
        response = requests.get("https://api.binance.com/api/v3/ticker/bookTicker?symbol=USDTBRL")
        data = response.json()
        bid = float(data['bidPrice'])
        ask = float(data['askPrice'])
        self.logger.info(f"Orderbook | ask : {ask} | bid: {bid}")
        return {'bid': bid, 'ask': ask}

    def _get_observation(self, snapshot):
        bid = snapshot['bid']
        ask = snapshot['ask']
        spread_percentage = (ask - bid) / bid * 100
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
            self.logger.info(f"🟢 BUY: {usd:.2f} USDT @ {bid:.2f} ARS | Fee Incluido | Cash: {self.cash:.2f}, Inv: {self.inventory:.2f}, Eq: {equity:.2f}")

    def ejecutar_venta(self, ask, usd):
        if self.inventory >= usd:
            self.inventory -= usd
            ars_received = usd * ask * (1 - self.fee_rate)
            self.cash += ars_received

            buy_trades = [t for t in self.trades if t[0] == "BUY"]
            if buy_trades:
                avg_buy_price = sum(t[1] * t[2] for t in buy_trades) / sum(t[1] for t in buy_trades)
                pnl = (ask - avg_buy_price) * usd - (usd * ask * self.fee_rate + usd * avg_buy_price * self.fee_rate)
            else:
                pnl = 0.0

            equity = self.calcular_equity(ask)
            self.trades.append(("SELL", usd, ask, pnl))
            self.logger.info(f"🔴 SELL: {usd:.2f} USDT @ {ask:.2f} ARS | Fee Incluido | PnL: {pnl:.2f}, Cash: {self.cash:.2f}, Inv: {self.inventory:.2f}, Eq: {equity:.2f}")

    def calcular_equity(self, bid):
        return self.cash + self.inventory * bid

    def run(self):
        while True:
            snapshot = self.get_orderbook_snapshot()
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

            time.sleep(1)

# Ejemplo de uso:
# bot = PaperTradingBot(model_path="models/model_equity_2269.zip")
# bot.run()
