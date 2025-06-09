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
├── optuna_final_results.csv
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

## 📬 Contacto

Proyecto creado y entrenado por [Tu Nombre o Usuario].
