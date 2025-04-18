import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from BlackSchole import *
from BatesMCModified import *

S0 = 100
T = 1.0
r = 0.05
v0 = 0.04
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.5
M = 10000
N = 200
strikes = np.linspace(100, 180, 150)

param_variants = {
    '$\\xi_J$': np.linspace(.1, .8, 3),
    '$\\sigma_J$': np.linspace(.1, .7, 3),
    '$a_\\mu$': np.linspace(-.5, .5, 3),
    '$b_\\mu$': np.linspace(.01, .1, 3),
    '$c_\\mu$': np.linspace(-.1, .1, 3),
    '$d_\\mu$': np.linspace(1, 4, 3)
}

base_params = {
    '$\\xi_J$': 0.3,
    '$\\sigma_J$': 0.2,
    '$a_\\mu$': -0.2,
    '$b_\\mu$': 0.02,
    '$c_\\mu$': 0.0,
    '$d_\\mu$': 2
}

fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs = axs.flatten()

for i, (pname, plist) in enumerate(param_variants.items()):
    print(i)
    for val in plist:
        params = base_params.copy()
        params[pname] = val
        ivs = []
        for K in strikes:

            price = monte_carlo_bates(S0, K, T, r, v0, kappa, theta, sigma, rho,
                                      params['$\\xi_J$'], params['$\\sigma_J$'], 'call', M, N,
                                      params['$a_\\mu$'], params['$b_\\mu$'], params['$c_\\mu$'], params['$d_\\mu$'])
            iv = BisecBSIV('C', S0, K, r, 0.0, T, 0.001, 3.0, price, 1e-3, 1000)
            ivs.append(iv)
        axs[i].plot(strikes, ivs, label=f"{pname}={val:.2f}")
    axs[i].set_title(f"Impact of {pname} on IV")
    axs[i].set_xlabel("Strike")
    axs[i].set_ylabel("Implied Volatility")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
