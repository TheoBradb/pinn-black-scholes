import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="PINN Option Pricer", layout="wide")

class OptionPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, S, t, K, r, sigma):
        x = torch.cat([S / 100.0, t, K / 100.0, r / 0.05, sigma / 0.20], dim=1)
        return self.net(x) * 100.0

@st.cache_resource
def get_model():
    m = OptionPINN()
    m.load_state_dict(torch.load("universal_pinn.pth", map_location="cpu"))
    m.eval()
    return m

model = get_model()

def bs_call(S, K, T, t, r, sigma):
    tau = T - t
    if tau <= 1e-6:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

st.sidebar.title("Parameters")
S_val = st.sidebar.slider("Stock Price (S)", 50.0, 180.0, 100.0, 1.0)
t_val = st.sidebar.slider("Evaluation Time (t)", 0.0, 0.99, 0.0, 0.02)
K_val = st.sidebar.slider("Strike Price (K)", 70.0, 130.0, 100.0, 1.0)
r_val = st.sidebar.slider("Risk-free Rate (r)", 0.01, 0.08, 0.05, 0.005)
sig_val = st.sidebar.slider("Volatility (σ)", 0.10, 0.35, 0.20, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Theo Bradbury** · [GitHub](https://github.com/TheoBradb)")

S_t = torch.tensor([[S_val]], dtype=torch.float32, requires_grad=True)
t_t = torch.tensor([[t_val]], dtype=torch.float32, requires_grad=True)
K_t = torch.tensor([[K_val]], dtype=torch.float32)
r_t = torch.tensor([[r_val]], dtype=torch.float32)
sig_t = torch.tensor([[sig_val]], dtype=torch.float32, requires_grad=True)

pred = model(S_t, t_t, K_t, r_t, sig_t)

V_S, V_t, V_sig = torch.autograd.grad(pred, (S_t, t_t, sig_t), create_graph=True)
V_SS = torch.autograd.grad(V_S, S_t)[0]

pinn_price = pred.item()
exact_price = bs_call(S_val, K_val, 1.0, t_val, r_val, sig_val)
diff = abs(pinn_price - exact_price)

st.title("Universal PINN Option Pricing Engine")
st.caption("Developed by **Theo** | [GitHub Profile](https://github.com/TheoBradb)")

c1, c2, c3 = st.columns(3)
c1.metric("PINN Price", f"${pinn_price:.2f}")
c2.metric("Exact Black-Scholes", f"${exact_price:.2f}")
c3.metric("Abs Error", f"${diff:.2f}")

st.write("---")
st.subheader("Greeks via Autograd")

g1, g2, g3, g4 = st.columns(4)
g1.metric("Delta (Δ)", f"{V_S.item():.4f}")
g2.metric("Gamma (Γ)", f"{V_SS.item():.4f}")
g3.metric("Theta (Θ)", f"{V_t.item():.4f}")
g4.metric("Vega (ν)", f"{V_sig.item():.4f}")

st.write("---")
st.subheader("3D Solution Surface")

S_points = np.linspace(60, 160, 30)
t_points = np.linspace(0, 0.99, 30)
S_grid, t_grid = np.meshgrid(S_points, t_points)

S_in = torch.tensor(S_grid.flatten()[:, None], dtype=torch.float32)
t_in = torch.tensor(t_grid.flatten()[:, None], dtype=torch.float32)
K_in = torch.full_like(S_in, K_val)
r_in = torch.full_like(S_in, r_val)
sig_in = torch.full_like(S_in, sig_val)

with torch.no_grad():
    Z = model(S_in, t_in, K_in, r_in, sig_in).numpy().reshape(30, 30)

fig = go.Figure(data=[go.Surface(z=Z, x=S_points, y=t_points, colorscale="plasma")])
fig.update_layout(
    scene=dict(
        xaxis_title="Stock Price (S)",
        yaxis_title="Time (t)",
        zaxis_title="Price (V)"
    ),
    margin=dict(l=10, r=10, b=10, t=30),
    height=550
)

st.plotly_chart(fig, use_container_width=True)