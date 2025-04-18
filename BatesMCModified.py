import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def muFunc(S, a_mu, b_mu, c_mu, d_mu):
    return a_mu * np.tanh(b_mu * S - d_mu) + c_mu

def lambdaFunc(S, a_lam, b_lam):
    return a_lam + b_lam * S


def bates_model(S0, v0, r, kappa, theta, sigma, rho, lambda_, sigma_j, T, N, M, a_mu, b_mu, c_mu, d_mu):
    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    jumplist = {}

    for i in range(1, N + 1):
        dW1 = np.random.normal(0, np.sqrt(dt), M)
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), M)

        dN = np.random.poisson(lambda_ * dt, M) # dN = np.random.poisson(lambdaFunc(S[:, i - 1]) * dt, M)
        J = np.random.normal(muFunc(S[:, i - 1], a_mu, b_mu, c_mu, d_mu), sigma_j, M) * dN # J = np.random.normal(muFunc(S[:, i-1], a_mu, b_mu, c_mu), sigma_j, M) * dN
        nonzero_indices = np.nonzero(dN)[0]
        if len(nonzero_indices) != 0:
            for idx in nonzero_indices:
                jumplist[(i, idx)] = (dN[idx], J[idx])

        v[:, i] = v[:, i - 1] + kappa * (theta - v[:, i - 1]) * dt + sigma * np.sqrt(v[:, i - 1]) * dW2
        v[:, i] = np.maximum(v[:, i], 0)  # Ensure non-negative volatility

        S[:, i] = S[:, i - 1] * np.exp(
            (r - 0.5 * v[:, i - 1] - lambda_ * (np.exp(muFunc(S[:, i - 1], a_mu, b_mu, c_mu, d_mu) + 0.5 * sigma_j ** 2) - 1)) * dt
             + np.sqrt(v[:, i - 1]) * dW1
             + J
        )

    return S, v, jumplist

def monte_carlo_bates(S0, K, T, r, v0, kappa, theta, sigma, rho, lambda_, sigma_j, option_type, M,
                       N, a_mu, b_mu, c_mu, d_mu):

    S, _, _ = bates_model(S0, v0, r, kappa, theta, sigma, rho, lambda_, sigma_j, T, N, M, a_mu, b_mu, c_mu, d_mu)

    # Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(S[:, -1] - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - S[:, -1], 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price

