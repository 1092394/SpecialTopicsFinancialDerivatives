import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

from HestonGL import HestonPriceGaussLaguerre
from HestonFFT import HestonCallFFT
from BlackSchole import BisecBSIV

S0 = 100
r = 0.05
q = 0.01
kappa = 2.0
theta = 0.05
sigma = 0.3
v0 = 0.05
rho = 0.45
lam = 0

trap = 1
alpha = 1.5
uplimit = 600
rule = 'T'
PutCall = 'C'

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
})

n_values = np.arange(2, 40)
prices = [HestonPriceGaussLaguerre(PutCall, S0, S0, 1.5, r, q,
                                   kappa, theta, sigma, lam, v0, rho, trap, n)
          for n in n_values]

GL_price = HestonPriceGaussLaguerre(PutCall, S0, S0, 1.5, r, q,
                                     kappa, theta, sigma, lam, v0, rho, trap, 32)
N_values = np.linspace(10, 500, 500)
fft_errors = []
for N in N_values:
    N = int(N)
    CallFFT, K_grid, *_ = HestonCallFFT(N, uplimit, S0, r, q, 1.5,
                                        kappa, theta, lam, rho, sigma, v0,
                                        trap, alpha, rule)
    price = float(interp1d(K_grid, CallFFT, kind='cubic', fill_value="extrapolate")(S0))
    fft_errors.append(abs(price - GL_price))

strikes = np.linspace(20, 130, 100)
taus = np.linspace(50, 250, 40)
K_grid, T_grid = np.meshgrid(strikes, taus, indexing='ij')
T_vals = T_grid / 365
IV_surface = np.zeros_like(T_vals)

for j in range(len(taus)):
    tau = taus[j] / 365
    CallFFT, K_fft, *_ = HestonCallFFT(
        N=1024, uplimit=600, S0=S0, r=r, q=q, tau=tau,
        kappa=kappa, theta=theta, lam=lam, rho=rho,
        sigma=sigma, v0=v0, Trap=trap, alpha=alpha, rule=rule
    )
    prices_iv = interp1d(K_fft, CallFFT, kind='cubic', fill_value="extrapolate")(strikes)
    for i, K_val in enumerate(strikes):
        IV_surface[i, j] = BisecBSIV('C', S0, K_val, r, q, tau, 0.001, 3.0, prices_iv[i], 1e-6, 1000)


fig = plt.figure(figsize=(21, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.6])


ax1 = fig.add_subplot(gs[0])
ax1.plot(n_values, prices, 'o-', color='steelblue')
ax1.set_xlabel("Gauss-Laguerre Order $n$")
ax1.set_ylabel("Option Price")
ax1.set_title("Price vs Gauss-Laguerre Polynomial Order $n$")
ax1.grid(True)


ax2 = fig.add_subplot(gs[1])
ax2.plot(N_values, fft_errors, 's--', color='firebrick')
ax2.set_xlabel("FFT Grid Size $N$")
ax2.set_ylabel("Absolute Error")
ax2.set_title(f"FFT Abs Error ($\\alpha$ = {alpha})")
ax2.grid(True)


ax3 = fig.add_subplot(gs[2], projection='3d')
surf = ax3.plot_surface(K_grid, T_grid, IV_surface, cmap='viridis',
                        edgecolor='none', linewidth=0, antialiased=True, alpha=0.8)
ax3.set_xlabel("Strike $K$")
ax3.set_ylabel("Maturity $\\tau$ (days)")
ax3.set_zlabel("IV (%)")
ax3.set_title("Heston Implied Volatility Surface")

plt.tight_layout()
plt.show()
