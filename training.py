import torch
import torch.nn as nn

# 1. THE NEURAL NETWORK (2 inputs -> 1 output)
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
        return self.net(x) * 100.0 # converting back to usd

def compute_pde_loss(model, S, t, K, r, sigma):
    S.requires_grad_(True) # Saves graph of s/t so automatic differentiation can take place 
    t.requires_grad_(True)

    V = model(S, t, K, r, sigma)
 
    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]
    

    residual = V_t + 0.5 * (sigma**2) * (S**2) * V_SS + r * S * V_S - r * V
    
    # The loss is how far this residual is from zero
    return torch.mean(residual**2)

def compute_adapative_lambda(model, loss_pde, loss_term):  # dynamic loss weighting
    grad_pde = torch.autograd.grad(loss_pde, model.parameters(), retain_graph = True) 
    grad_term = torch.autograd.grad(loss_term, model.parameters(), retain_graph=True)

    flattened_pde = torch.cat([g.contiguous().view(-1) for g in grad_pde])
    flattened_term = torch.cat([g.contiguous().view(-1) for g in grad_term])

    max_pde = torch.max(torch.abs(flattened_pde))
    mean_term = torch.mean(torch.abs(flattened_term))

    adaptive_lambda = (max_pde / (mean_term + 1e-8)).item()

    return adaptive_lambda

model = OptionPINN()
dummy_S = torch.rand(100, 1) * 100 + 50
dummy_t = torch.rand(100, 1)
dummy_K = torch.rand(100, 1) * 60.0 + 70.0
dummy_r = torch.rand(100, 1) * 0.07 + 0.01
dummy_sigma = torch.rand(100, 1) * 0.25 + 0.10


loss = compute_pde_loss(model, dummy_S, dummy_t, dummy_K, dummy_r, dummy_sigma)
print(f"Initial PDE Residual Loss (before training): {loss.item():.4f}")



model = OptionPINN()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.001)
lambda_term = 25.0 

for epoch in range(5001):
    optimiser.zero_grad()
    # --- A. Interior Points (All 5 variables sampled) ---
    s_int     = torch.rand(10000, 1) * 200.0 
    t_int     = torch.rand(10000, 1) * 1.0
    K_int     = torch.rand(10000, 1) * 60.0 + 70.0
    r_int     = torch.rand(10000, 1) * 0.07 + 0.01
    sigma_int = torch.rand(10000, 1) * 0.25 + 0.10
    loss_pde = compute_pde_loss(model, s_int, t_int, K_int, r_int, sigma_int)

    # --- B. Terminal Payoff Condition ---
    S_term     = torch.rand(10000, 1) * 200.0
    t_term     = torch.ones_like(S_term) * 1.0
    K_term     = torch.rand(10000, 1) * 60.0 + 70.0
    r_term     = torch.rand(10000, 1) * 0.07 + 0.01 
    sigma_term = torch.rand(10000, 1) * 0.25 + 0.10
    true_payoff = torch.clamp(S_term - K_term, min=0.0)
    V_term = model(S_term, t_term, K_term, r_term, sigma_term)
    loss_term = torch.mean((V_term - true_payoff)**2)

    # --- C. Boundary Condition (S = 0) ---
    S_zero     = torch.zeros(400, 1)
    t_zero     = torch.rand(400, 1) * 1.0
    K_zero     = torch.rand(400, 1) * 60.0 + 70.0 
    r_zero     = torch.rand(400, 1) * 0.07 + 0.01 
    sigma_zero = torch.rand(400, 1) * 0.25 + 0.10
    V_zero = model(S_zero, t_zero, K_zero, r_zero, sigma_zero)
    loss_zero = torch.mean(V_zero**2)


    if epoch % 100 ==0 and epoch != 0 : 
        new_lambda = compute_adapative_lambda(model, loss_pde, loss_term)

        new_lambda = max(5.0, min(new_lambda, 50.0))
        
        lambda_term = 0.9 * lambda_term + 0.1 * new_lambda #smoothens out changes in lambda 
        print(f"Epoch {epoch:4d} | Auto Tuned Lambda: {lambda_term:.2f}")

    # --- D. Optimize ---
    total_loss = loss_pde + lambda_term * loss_term + loss_zero
    total_loss.backward()
    optimiser.step()

    if epoch % 250 == 0: 
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():8.2f} | PDE Loss: {loss_pde.item():6.2f} | Payoff Loss: {loss_term.item():6.2f}")
  

print("\nTraining Complete")


test_S     = torch.tensor([[100.0]])
test_t     = torch.tensor([[0.0]])
test_K     = torch.tensor([[100.0]])
test_r     = torch.tensor([[0.05]])
test_sigma = torch.tensor([[0.20]])

model.eval()
with torch.no_grad():
    predicted_price = model(test_S, test_t, test_K, test_r, test_sigma).item()

print(f"\n--- BENCHMARK RESULTS ---")
print(f"PINN Predicted Option Price: ${predicted_price:.2f}")
print(f"Exact Black-Scholes Formula: $10.45")
print(f"Difference:                  ${abs(predicted_price - 10.45):.2f}")

torch.save(model.state_dict(), "universal_pinn.pth")