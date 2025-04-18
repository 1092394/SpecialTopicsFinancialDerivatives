import numpy as np

def HestonCallEuroOptPriceMC(S0, v0, K, r, q, T, kappa, vbar, sigma, rho,  M=100000, N=500, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    S = np.full(M, S0, dtype=np.float64)
    V = np.full(M, v0, dtype=np.float64)

    for _ in range(N):
        Z1 = np.random.randn(M)
        Z2 = np.random.randn(M)
        dW_S = np.sqrt(dt) * Z1
        dW_V = rho * np.sqrt(dt) * Z1 + np.sqrt(1 - rho ** 2) * np.sqrt(dt) * Z2
        V_current = np.maximum(V, 0)
        V = V + kappa * (vbar - V_current) * dt + sigma * np.sqrt(V_current) * dW_V
        V = np.maximum(V, 0)
        S = S * np.exp((r - q - 0.5 * V_current) * dt + np.sqrt(V_current) * dW_S)

    payoffs = np.maximum(S - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)

    return call_price

def HestonMCPathSim(S0, v0, r, q, T, kappa, vbar, sigma, rho,  M=100000, N=500, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    S = np.full(M, S0, dtype=np.float64)
    V = np.full(M, v0, dtype=np.float64)
    Slist = np.zeros([M, N + 1], dtype=np.float64)
    Vlist = np.zeros([M, N + 1], dtype=np.float64)
    Slist[:, 0] = S
    Vlist[:, 0] = V

    for _ in range(N):
        Z1 = np.random.randn(M)
        Z2 = np.random.randn(M)
        dW_S = np.sqrt(dt) * Z1
        dW_V = rho * np.sqrt(dt) * Z1 + np.sqrt(1 - rho ** 2) * np.sqrt(dt) * Z2
        V_current = np.maximum(V, 0)
        V = V + kappa * (vbar - V_current) * dt + sigma * np.sqrt(V_current) * dW_V
        V = np.maximum(V, 0)
        Vlist[:, _ + 1] = V

        S = S * np.exp((r - q - 0.5 * V_current) * dt + np.sqrt(V_current) * dW_S)
        Slist[:, _ + 1] = S
    return Slist, Vlist

# price = HestonCallEuroOptPriceMC(100, 0.05, 100, 0.05, 0.01, 1.5, 2., 0.05, 0.3, 0.45)
# print(price)