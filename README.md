# Universal Physics-Informed Neural Network (PINN) for Option Pricing

A continuous 5-dimensional deep learning model implemented in PyTorch that solves the Black-Scholes Partial Differential Equation (PDE) without real-world market training data.

The network models option price $V(S, t, K, r, \sigma)$ across continuous strike, expiration, risk-free rate, and implied volatility spaces, and extracts exact analytical risk sensitivities (Greeks: $\Delta, \Gamma, \Theta, \nu$) via automatic differentiation.

## Performance Benchmark

Trained with continuous collocation sampling and dynamic loss balancing:

| Metric | PINN Model | Analytical Black-Scholes | Absolute Error |
| :--- | :--- | :--- | :--- |
| **At-The-Money Call ($S=100, K=100, t=0$)** | **$10.46** | **$10.45** | **$0.01 (<0.1%)** |
| **Delta ($\Delta$)** | **0.6368** | **0.6368** | **Exact match (4 d.p.)** |
| **Inference Latency** | **< 1 ms** | Analytical | **Real-Time** |

## Features

- **Mesh-Free Solver:** Evaluates continuous coordinates without rigid finite difference grids.
- **Universal Parametric Architecture:** 5 input neurons ($S, t, K, r, \sigma$) enabling instant evaluation of arbitrary contracts without retraining.
- **Autograd Risk Sensitivities:** Continuous calculation of Delta, Gamma, Theta, and Vega directly from the PyTorch computational graph.
- **Interactive Web Interface:** Streamlit dashboard with 3D solution surface rendering via Plotly.

## Installation & Running Locally

```bash
git clone https://github.com/TheoBradb/pinn-black-scholes.git
cd pinn-black-scholes
pip install -r requirements.txt
streamlit run app.py
```
