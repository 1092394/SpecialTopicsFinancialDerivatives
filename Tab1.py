import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d

from HestonGL import HestonPriceGaussLaguerre
from HestonMC import HestonCallEuroOptPriceMC
from HestonFFT import HestonCallFFT

S0 = 100.0
r = 0.05
q = 0.01
T = 1.5
kappa = 2.0
theta = 0.05
sigma = 0.3
v0 = 0.05
rho = 0.45
lam = 0
trap = 1
n_laguerre = 32
MC_paths = 20000
MC_steps = 500

K_values = [80, 90, 100, 110, 120, 150]

results = []

for K in K_values:
    start = time.time()
    price_gl = HestonPriceGaussLaguerre('C', S0, K, T, r, q, kappa, theta, sigma,
                                        lam, v0, rho, trap, n_laguerre)
    time_gl = time.time() - start

    start = time.time()
    CallFFT, K_fft, *_ = HestonCallFFT(4096, 500, S0, r, q, T, kappa, theta,
                                       lam, rho, sigma, v0, trap, 1.5, 'T')
    price_fft = float(interp1d(K_fft, CallFFT, kind='cubic', fill_value='extrapolate')(K))
    time_fft = time.time() - start

    start = time.time()
    mc_vals = [HestonCallEuroOptPriceMC(S0, v0, K, r, q, T, kappa, theta, sigma,
                                        rho, M=MC_paths, N=MC_steps) for _ in range(3)]
    price_mc = np.mean(mc_vals)
    std_mc = np.std(mc_vals)
    time_mc = time.time() - start

    results.append({
        "Strike K": K,
        "GL Price": price_gl,
        "FFT Price": price_fft,
        "MC Price": price_mc,
        "MC StdErr": std_mc,
        "Abs Diff (GL - FFT)": abs(price_gl - price_fft),
        "Abs Diff (GL - MC)": abs(price_gl - price_mc),
        "Time GL (s)": time_gl,
        "Time FFT (s)": time_fft,
        "Time MC (s)": time_mc
    })

df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
print(df)
