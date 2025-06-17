# 🧠 Trading Strategy Optimization with PPO + Optuna

Este proyecto utiliza **Aprendizaje por Refuerzo (RL)** con el algoritmo **PPO (Proximal Policy Optimization)** de la librería `stable-baselines3`, aplicado a un entorno personalizado de trading. La búsqueda de los mejores parámetros de recompensa se realiza automáticamente con **Optuna**, una potente herramienta de optimización bayesiana.

## 📈 Objetivo

Entrenar un agente que aprenda a comprar, vender o mantenerse inactivo en función de datos de snapshots del orderbook, maximizando su **equity final** bajo condiciones de mercado reales.

## 🧰 Estructura del Proyecto

```
project/
├── data/
├── models/
├── models_final/
├── env_simple.py
├── optimize_refined.py
├── optimize_final.py
├── evaluate_best_configs.py
├── evaluate_with_metrics.py
└── README.md
```

## ⚙️ Tecnologías

- [Stable-Baselines3 (PPO)](https://github.com/DLR-RM/stable-baselines3)
- [Optuna](https://optuna.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Pandas, NumPy]

## 🚀 ¿Cómo correr la optimización?

### 1. Instalá dependencias

```bash
pip install stable-baselines3 optuna gym pandas numpy
```

### 2. Ejecutá una optimización refinada

```bash
python optimize_final.py
```

### 3. Evaluar consistencia de los mejores modelos

```bash
python evaluate_best_configs.py
```

## 📊 Dashboard en tiempo real

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///final_optuna_study.db
```

Abrí en tu navegador:

```
http://localhost:8080
```

## 🧠 ¿Qué se optimiza?

- `reward_trade`
- `reward_hold`
- `reward_profit`
- `reward_loss`
- `reward_idle`

El entorno también usa el coeficiente de Hurst, el exponente de Lyapunov y volatilidad local.

## 📊 Paper Trading

Una vez que tengas modelos entrenados, podés probar su rendimiento en condiciones reales usando los módulos de **paper trading**. El proyecto incluye dos modalidades:

### 🔴 Paper Trading en Vivo (`paper_trading.py`)

Ejecuta trading en tiempo real conectándose a la API de Binance para obtener datos del orderbook.

**Características:**
- Conexión en tiempo real a Binance API
- Capital inicial: 5000 Moneda quote
- Fee de trading: 0.1% (como Binance)
- Logging completo de operaciones
- Cálculo de PnL y equity en tiempo real

**Uso:**
```python
from paper_trading import PaperTradingBot

# Cargar un modelo entrenado
bot = PaperTradingBot(model_path="models/model_equity_2269.zip")

# Iniciar trading en vivo
bot.run()
```

**Lógica de Trading:**
- **Acción 0**: HOLD (mantener posición)
- **Acción 1**: BUY (comprar 5 Moneda base)
- **Acción 2**: SELL (vender inventario completo si ≥20 Moneda base, sino vender 5 Moneda base)

### 🟡 Paper Trading con Datos Históricos (`paper_trading_mocked.py`)

Simula trading usando datos históricos desde archivos CSV, ideal para backtesting y evaluación controlada.

**Características:**
- Usa datos históricos del orderbook
- Capital inicial: 100 Moneda quote (para pruebas)
- Procesa datos secuencialmente desde CSV
- Misma lógica de trading que la versión en vivo
- Termina automáticamente al finalizar los datos

**Uso:**
```python
from paper_trading_mocked import PaperTradingBot, MockedBinance

# Configurar exchange simulado
exchange = MockedBinance("data/BINANCE-USDT_BRL-100_depth-1749231790356.csv")

# Cargar modelo
bot = PaperTradingBot(model_path="models/model_equity_2269.zip")

# Ejecutar backtesting
bot.run(exchange)
```

### 📈 Métricas y Logging

Ambos bots generan logs detallados que incluyen:

- **Operaciones**: Registro de cada compra/venta con precios y cantidades
- **PnL**: Profit and Loss de cada operación
- **Equity**: Valor total de la cartera (cash + inventario)
- **Fees**: Cálculo automático de comisiones
- **Timestamps**: Marca temporal de cada operación

**Ejemplo de log:**
```
2024-01-15 14:30:25,123 __main__ INFO 🟢 BUY: 5.00 USDT @ 5.45 ARS | Cash: 4972.75, Inv: 5.00, Eq: 5000.00
2024-01-15 14:31:30,456 __main__ INFO 🔴 SELL: 5.00 USDT @ 5.47 ARS | PnL: 0.08, Cash: 4999.89, Inv: 0.00, Eq: 4999.89
```

### 🔧 Configuración de Archivos

Los archivos CSV de datos históricos deben tener la estructura:
```csv
bid,ask,spread_percentage
5.45,5.47,0.37
5.46,5.48,0.36
...
```

### 💡 Casos de Uso

- **Trading en Vivo**: Para probar modelos en condiciones reales de mercado
- **Backtesting**: Para evaluar rendimiento histórico con datos conocidos
- **Comparación**: Evaluar múltiples modelos bajo las mismas condiciones
- **Desarrollo**: Probar cambios en la lógica de trading sin riesgo

## 📬 Contacto

Proyecto creado y entrenado por Blas Martin Castro.
