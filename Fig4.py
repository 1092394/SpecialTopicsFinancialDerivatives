import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from HestonFFT import HestonCallFFT
from BlackSchole import BisecBSIV

S0 = 100.0
r = 0.03
q = 0.00
T_smile = 1.0
lam = 0
trap = 1
alpha = 1.5
default_params = {
    'kappa': 2.0,
    'theta': 0.02,
    'sigma': 0.8,
    'v0': 0.01,
    'rho': -0.6,
}

params_to_plot = ['r', 'rho', 'sigma', 'v0', 'theta', 'kappa']
param_ranges = {
    'r' :     np.linspace(0.005, 0.07,15),
    'rho':    np.linspace(-0.95, 0.5, 15),
    'sigma':  np.linspace(0.2, 1.2, 15),
    'v0':     np.linspace(0.005, 0.1, 15),
    'theta':  np.linspace(0.01, 0.5, 15),
    'kappa':  np.linspace(0.5, 4.0, 15),
}
label_map = {
    'r' : 'Risk-free Rate $r$',
    'rho': 'Correlation $\\rho$',
    'sigma': 'Vol-of-Vol $\\sigma$',
    'v0': 'Initial Variance $v_0$',
    'theta': 'Long-term Variance $\\bar{v}$',
    'kappa': 'Mean Reversion $\\kappa$',
}

strikes_smile = np.linspace(70, 130, 40)

fig = plt.figure(figsize=(18, 10))
for idx, param_name in enumerate(params_to_plot, start=1):
    param_values = param_ranges[param_name]
    iv_smile_matrix = []

    for param_val in param_values:
        p = default_params.copy()
        p[param_name] = param_val

        CallFFT, K_fft, _, _ = HestonCallFFT(
            N=1024, uplimit=600, S0=S0, r=r, q=q, tau=T_smile,
            kappa=p['kappa'], theta=p['theta'], sigma=p['sigma'],
            v0=p['v0'], rho=p['rho'], lam=lam,
            Trap=trap, alpha=alpha, rule='T'
        )
        interp_fn = interp1d(K_fft, CallFFT, kind='cubic', fill_value="extrapolate")
        prices = interp_fn(strikes_smile)

        ivs = [BisecBSIV('C', S0, K, r, q, T_smile, 0.001, 3.0, price, 1e-6, 1000)
               for K, price in zip(strikes_smile, prices)]
        iv_smile_matrix.append(ivs)

    PARAM_mesh, STRIKE_mesh = np.meshgrid(param_values, strikes_smile, indexing='ij')
    IV_mesh = np.array(iv_smile_matrix)

    ax = fig.add_subplot(2, 3, idx, projection='3d')
    surf = ax.plot_surface(STRIKE_mesh, PARAM_mesh, IV_mesh,
                           cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.75)
    ax.set_title(label_map[param_name])
    ax.set_xlabel("Strike $K$")
    ax.set_ylabel(label_map[param_name])
    ax.set_zlabel("IV")

fig.suptitle("Implied Volatility Smile Sensitivity to Heston Parameters (T = 1.0)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
