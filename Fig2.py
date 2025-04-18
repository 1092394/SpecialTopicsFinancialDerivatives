from HestonGL import *
from HestonMC import *
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
})


S0, K = 100, 100
r, q = 0.05, 0.01
T = 1.5
kappa = 1.0
vbar = 0.2
sigma = 0.1
v0 = 0.05
rho = 0.2

M_sample = 600
N_sample = 15
# np.random.seed(20)
seed = 25

S_paths, v_paths = HestonMCPathSim(S0, v0, r, q, T, kappa, vbar, sigma, rho,  M=N_sample, N=M_sample, seed=seed)

from tqdm import tqdm
M_values = 10 ** np.linspace(0, 5, 50)#[1, 1000, 5000, 10000, 50000, 100000]
M_values = np.round(M_values).astype(int)
mc_prices = []
mc_std_errs = []
repeats = 5

GL_price = HestonPriceGaussLaguerre('C', S0, K, T, r, q, kappa, vbar, sigma, 0, v0, rho, 1, 32)

for M in tqdm(M_values):
    results = [HestonCallEuroOptPriceMC(S0, v0, K, r, q, T, kappa, vbar, sigma, rho, M=M, N=500) for _ in range(repeats)]
    mc_prices.append(np.mean(results))
    mc_std_errs.append(np.std(results) / np.sqrt(repeats))

logM = np.log10(M_values)
price_errors = np.array(mc_prices) - GL_price

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(24, 11))
outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.3)

left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[1, 1], hspace=0.3)

ax_s = plt.Subplot(fig, left[0])
for i in range(N_sample):
    ax_s.plot(np.linspace(0, T, M_sample + 1), S_paths[i], alpha=0.8)
ax_s.set_title("Simulated Asset Price Paths $S_t$")
ax_s.set_xlabel("Time")
ax_s.set_ylabel("$S_t$")
ax_s.grid(True)
fig.add_subplot(ax_s)

ax_v = plt.Subplot(fig, left[1])
for i in range(N_sample):
    ax_v.plot(np.linspace(0, T, M_sample + 1), v_paths[i], alpha=0.8)

ax_v.set_title("Simulated Variance Paths $v_t$")
ax_v.set_xlabel("Time")
ax_v.set_ylabel("$v_t$")
ax_v.grid(True)
fig.add_subplot(ax_v)


ax_right = fig.add_subplot(outer[1])
ax_right.plot(logM, price_errors, 'o-', label='MC Price - GL Price')
ax_right.plot(logM, mc_std_errs, 's--', label='MC Std. Error')
ax_right.set_title("MC Price Error and Std. Error vs Number of Paths")
ax_right.set_xlabel("log10(Number of MC Paths)")
ax_right.set_ylabel("Error / Std. Error")
ax_right.grid(True)
ax_right.legend()

plt.tight_layout()
plt.show()
