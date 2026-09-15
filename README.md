

# Universal Physics-Informed Neural Network (PINN) for Option Pricing

[Live Preview](https://theobradbury-pinn-black-scholes.streamlit.app)

A continuous 5-dimensional deep learning model implemented in PyTorch that solves the **Black-Scholes Partial Differential Equation (PDE)** without requiring supervised training data or real-world option prices.

Rather than training on observed quotes or pre-calculated analytical prices, the network is trained as a **continuous mesh-free solver** by enforcing the governing PDE, terminal payoff condition, and boundary conditions directly through the loss function using automatic differentiation.

The model represents option value as a continuous function across a 5D parameter space:

$$V = V(S, t, K, r, \sigma)$$

where:

- $S$ — underlying asset price ($S \in [0, 200]$)
- $t$ — evaluation time ($t \in [0, 1.0]$, with maturity at $T = 1.0$)
- $K$ — strike price ($K \in [70, 130]$)
- $r$ — risk-free interest rate ($r \in [0.01, 0.08]$)
- $\sigma$ — volatility ($\sigma \in [0.10, 0.35]$)

This formulation allows a single trained network to price European call options across continuous strikes, maturities, and market conditions without retraining.

---

## Mathematical Formulation

For a European call option, the governing Black-Scholes PDE is:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

subject to the **terminal payoff condition** at maturity ($T = 1.0$):

$$V(S, T, K, r, \sigma) = \max(S - K, 0)$$

and the **lower boundary condition** at $S = 0$:

$$V(0, t, K, r, \sigma) = 0$$

The neural network acts as a continuous function approximator $V_\theta \approx V$:

$$V_\theta(S, t, K, r, \sigma) \approx V(S, t, K, r, \sigma)$$

where $\theta$ represents the trainable weights and biases of the network.

During training, **PyTorch automatic differentiation (`torch.autograd`)** computes the required spatial and temporal partial derivatives directly from the computational graph:

$$\mathcal{R}_\theta = \frac{\partial V_\theta}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V_\theta}{\partial S^2} + rS \frac{\partial V_\theta}{\partial S} - rV_\theta$$

The network is optimised to minimise this residual while simultaneously satisfying the terminal payoff and boundary constraints.

---

## Model Architecture & Normalization

The model is structured as a Fully Connected Multi-Layer Perceptron (MLP) with 3 hidden layers and 128 neurons per layer:

$$
\begin{aligned}
\text{Linear}(5 \to 128) &\to \text{Tanh} \to \text{Linear}(128 \to 128) \to \text{Tanh} \\
&\to \text{Linear}(128 \to 128) \to \text{Tanh} \to \text{Linear}(128 \to 1)
\end{aligned}
$$

### Input Normalization & Output Scaling

Because inputs have substantially different numerical ranges (e.g. $S \sim 100$ vs $r \sim 0.05$), the input features are normalised prior to the forward pass, and the network output is rescaled back to dollar units:

$$\mathbf{x}_{\text{norm}} = \left[ \frac{S}{100}, \; t, \; \frac{K}{100}, \; \frac{r}{0.05}, \; \frac{\sigma}{0.20} \right]$$

$$V_\theta(S, t, K, r, \sigma) = \text{net}(\mathbf{x}_{\text{norm}}) \times 100.0$$

---

## Training Procedure

The network is trained using collocation points uniformly sampled across the parameter domain at every epoch:

- **10,000 interior points:** Sampled across the full 5D domain 
- **10,000 terminal points:** Sampled at $t = 1.0$ 
- **400 boundary points:** Sampled at $S = 0$ 

### Dynamic Loss Balancing


To prevent the terminal condition loss from overpowering or lagging behind the PDE loss, the terminal weight  is dynamically adjusted every 100 epochs



Training runs for 5,000 epochs using the Adam optimiser 

---

## Automatic Differentiation & Greeks

Because the neural representation is inherently smooth and twice-differentiable, market risk sensitivities (the Greeks) are extracted directly from the computational graph via `torch.autograd` without finite-difference approximations:

### Delta ($\Delta$)

$$\Delta = \frac{\partial V_\theta}{\partial S}$$

Measures sensitivity to changes in the underlying asset price.

### Gamma ($\Gamma$)

$$\Gamma = \frac{\partial^2 V_\theta}{\partial S^2}$$

Second-order sensitivity with respect to the underlying price.

### Theta ($\Theta$)

$$\Theta = \frac{\partial V_\theta}{\partial t}$$

Rate of change of option value with respect to evaluation time $t$.

### Vega ($\nu$)

$$\nu = \frac{\partial V_\theta}{\partial \sigma}$$

Sensitivity of option value with respect to asset volatility.

---

## Performance Benchmark

The trained model (`final.pth`) was benchmarked against the analytical Black-Scholes formula at $S = 100$, $K = 100$, $t = 0.0$ (1 year to maturity), $r = 0.05$, and $\sigma = 0.20$:

| Metric | PINN Model | Analytical Black-Scholes | Absolute Error |
| :--- | :--- | :--- | :--- |
| **At-The-Money Call ($S=100, K=100, t=0$)** | **\$10.46** | **\$10.45** | **\$0.01 ($< 0.1\%$)** |
| **Delta ($\Delta$)** | **0.6368** | **0.6368** | **Exact match (4 d.p.)** |
| **Inference Latency** | **$< 1\text{ ms}$** | Analytical | **-** |

## Learning Resources & Video References

This project was built from scratch following these video lectures and tutorials:

### Machine Learning & Autograd Foundations

-  [3Blue1Brown — Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson *(Function composition, gradient descent, and backpropagation calculus)*

-  [Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) by Andrej Karpathy *(From-scratch walkthrough of automatic differentiation engines and computational graphs)*

### Physics-Informed Neural Networks (PINNs)

-  [Physics Informed Neural Networks (PINNs)](https://www.youtube.com/watch?v=-zrY7P2dVC4&t=1215s) by Prof. Steve Brunton, University of Washington *(Mesh-free continuous PDE loss formulation)*



### Mathematical Finance & Options

-  [MIT OpenCourseWare — Black-Scholes Formula & Risk-Neutral Valuation](https://www.youtube.com/watch?v=TnS8kI_KuJc)  *(MIT 18.S096: Topics in Mathematics with Applications in Finance)*

-  [Black Scholes PDE Derivation using Delta Hedging](https://www.youtube.com/watch?v=-qa2B_sCpZQ) by  quantpie *(Delta-hedged no-arbitrage PDE derivation)*

---

## Interactive Visualisation

The repository includes a Streamlit application (`app.py`) for interactive exploration:

- Real-time pricing and Greeks calculation via PyTorch autograd.
- Side-by-side verification against the closed-form Black-Scholes formula.
- Interactive 3D solution surface  rendered with Plotly.

---

