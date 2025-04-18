from zipfile import error

import numpy as np
from scipy.stats import norm
from HestonGL import HestonPriceGaussLaguerre
from scipy.optimize import minimize
import time
from Download_Data import *
from BlackSchole import *

def HestonObjFun(param, S, rf, q, MktPrice, K, T, MktIV):

    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    v0 = param[3]
    rho = param[4]
    lam = 0

    N = len(MktIV) # num of data
    if N != len(K) or N != len(T):
        raise ValueError("Input MktIV, K and T must have the same length.")

    errorIter = np.zeros(N)
    modelPrice = np.zeros(N)
    # modelIV = np.zeros(N)
    for _ in range(N):
        modelPrice[_] = HestonPriceGaussLaguerre('C', S, K[_], T[_], rf, q, kappa, theta, sigma, lam, v0, rho, 0, 32)

        d = (np.log(S / K[_]) + (rf - q + MktIV[_] ** 2 / 2) * T[_]) / (MktIV[_] * np.sqrt(T[_]))
        Vega = S * norm.pdf(d) * np.sqrt(T[_])
        errorIter[_] = (modelPrice[_] - MktPrice[_]) ** 2 / Vega ** 2

        # modelIV[_] = BisecBSIV('C', S, K[_], rf, q, T[_], a=0.001, b=5, MktPrice=modelPrice[_], Tol=1e-5, MaxIter=1000)
        # errorIter[_] = (MktIV[_] - modelIV[_]) ** 2

    return errorIter.sum() / N



def HestonCalibrate(MktIV, T, K, rf, q, S, MktPrice, start=None):
    if start is None:
        start = np.array([9.0, 0.05, 0.3, 0.05, -0.8])

    e = 1e-5
    lb = [e, e, e, e, -0.999]
    ub = [20, 2, 2, 2, 0.999]
    bounds = [(lb[i], ub[i]) for i in range(len(lb))]

    N = len(MktIV)
    if N != len(K) or N != len(T):
        raise ValueError("Input MktIV, K and T must have the same length.")

    def objective(p):
        loss = HestonObjFun(p, S, rf, q, MktPrice, K, T, MktIV)
        # print(f"Loss = {loss:.2f} for params: {p}")
        return loss

    start_time = time.time()
    res = minimize(objective, start, bounds=bounds, method='Nelder-Mead', options={'gtol': 1e-9, 'disp': True})
    elapsed = time.time() - start_time
    param = res.x
    print(res.values())

    lam = 0
    kappa, theta, sigma, v0, rho_val = param


    print("Calibrated  Parameters")
    print("kappa = {:.4f}, theta = {:.4f}, sigma = {:.4f}, v0 = {:.4f}, rho = {:.4f}, lambda = {:.4f}".format(
        kappa, theta, sigma, v0, rho_val, lam))
    print("Opt costs {:.4f} Sec".format(elapsed))

    return param, lam

if __name__ == '__main__':
    n = 3
    data, rf, q, S = returnData()
    import matplotlib.pyplot as plt

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs = axs.flatten()

    avg_errors = []

    for i in range(n):
        T, K, MktIV, MktPrice = data[i][0], data[i][1], data[i][2], data[i][3]
        param, lam = HestonCalibrate(MktIV, T, K, rf, q, S, MktPrice)
        kappa, theta, sigma, v0, rho = param

        modelPrice = np.array([max(1e-7,
            HestonPriceGaussLaguerre('C', S, K[j], T[0], rf, q, kappa, theta, sigma, lam, v0, rho, 1, 32))
            for j in range(len(K))
        ])
        modelIV = np.array([BisecBSIV('C', S, K[j], rf, q, T[0], a=0.001, b=20, MktPrice=modelPrice[j], Tol=1e-7, MaxIter=2000) for j in range(len(K))])

        # print(len(K))
        avg_error = np.mean(np.abs(modelIV - MktIV))
        avg_errors.append(avg_error)

        axs[i].plot(K, MktIV, 'ko-', label="Market IV")
        axs[i].plot(K, modelIV, 'r--', label="Heston IV")
        axs[i].set_title(f"Maturity: {round(T[0] * 365)} days")
        axs[i].set_xlabel("Strike")
        axs[i].set_ylabel("IV")
        axs[i].legend()
        axs[i].grid(True)

    # Last subplot: average error vs tau
    taus = [data[i][0][0] * 365 for i in range(n)]
    axs[n].plot(taus, avg_errors, 'b-o')
    axs[n].set_title("Average Absolute Error vs Maturity")
    axs[n].set_xlabel("Maturity (days)")
    axs[n].set_ylabel("Avg Abs Error")
    axs[n].grid(True)

    plt.tight_layout()
    plt.show()

