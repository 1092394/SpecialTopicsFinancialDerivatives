import numpy as np
import matplotlib.pyplot as plt
from BatesMCModified import bates_model, monte_carlo_bates
from HestonGL import HestonPriceGaussLaguerre

S0 = 100
T = 1.0
r = 0.05
v0 = 0.04
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.5
lambda_ = 0.6
sigma_j = 0.1
a_mu, b_mu, c_mu, d_mu = -0.8, 0.03, 0.1, 3

M = 5000
N = 300

np.random.seed(3)
S_paths, v_paths, jumplist = bates_model(S0, v0, r, kappa, theta, sigma, rho,
                                         lambda_, sigma_j, T, N, M, a_mu, b_mu, c_mu, d_mu)
t_grid = np.linspace(0, T, N + 1)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))


for i in range(5):
    axs[0].plot(t_grid, S_paths[i])

for (t_idx, path_idx), (dn_val, jump_val) in jumplist.items():
    if path_idx < 5:
        axs[0].annotate('', xy=(t_grid[t_idx], S_paths[path_idx, t_idx]),
                        xytext=(t_grid[t_idx], S_paths[path_idx, t_idx] + 5),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

axs[0].set_title("Paths with Jumps (Marked by Arrows)")
axs[0].set_xlabel("Time $t$")
axs[0].set_ylabel("Asset Price $S_t$")
axs[0].grid(True)


strikes = np.linspace(60, 140, 25)
bates_prices = []
heston_prices = []

for K in strikes:
    price_bates = monte_carlo_bates(S0, K, T, r, v0, kappa, theta, sigma, rho,
                                    lambda_, sigma_j, 'call', M, N, a_mu, b_mu, c_mu, d_mu)
    price_heston = HestonPriceGaussLaguerre('C', S0, K, T, r, 0.0,
                                             kappa, theta, sigma, 0.0, v0, rho, 1, 32)
    bates_prices.append(price_bates)
    heston_prices.append(price_heston)

axs[1].plot(strikes, heston_prices, label="Heston", color='blue')
axs[1].plot(strikes, bates_prices, label="with Jump", linestyle='--', color='red')
axs[1].set_title("Option Price: Heston vs Jump")
axs[1].set_xlabel("Strike $K$")
axs[1].set_ylabel("Call Price $V$")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
