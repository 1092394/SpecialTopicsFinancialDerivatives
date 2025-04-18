import numpy as np
import math


def GenerateGaussLaguerre(n):
    L = np.zeros(n + 1)
    for k in range(0, n + 1):
        L[k] = ((-1) ** k) / math.factorial(k) * math.comb(n, k)

    L = np.flip(L)

    x = np.roots(L)
    x = np.real(x)
    x = np.sort(x)

    w = np.zeros(n)
    for j in range(n):
        S = 0.0
        for k in range(1, n + 1):
            S += ((-1) ** k) / math.factorial(k - 1) * math.comb(n, k) * (x[j] ** (k - 1))
        w[j] = np.exp(x[j]) / (x[j] * (S ** 2))

    return x, w

def HestonProb(phi, kappa, theta, lam, rho, sigma, tau, K, S, r, q, v, Pnum, Trap):
    x = np.log(S)
    a = kappa * theta

    if Pnum == 1:
        u = 0.5
        b = kappa + lam - rho * sigma
    else:
        u = -0.5
        b = kappa + lam

    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if Trap == 1:
        c = 1.0 / g
        D = ((b - rho * sigma * 1j * phi - d) / sigma ** 2) * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * (((b - rho * sigma * 1j * phi - d) * tau) - 2 * np.log(G))
    elif Trap == 0:
        G = (1 - g * np.exp(d * tau)) / (1 - g)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * (((b - rho * sigma * 1j * phi + d) * tau) - 2 * np.log(G))
        D = ((b - rho * sigma * 1j * phi + d) / sigma ** 2) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    else:
        raise ValueError("Trap parameter must be either 1 (Little Trap) or 0 (Original Heston).")

    f = np.exp(C + D * v + 1j * phi * x)

    y = np.real(np.exp(-1j * phi * np.log(K)) * f / (1j * phi))

    return y


def HestonPriceGaussLaguerre(PutCall, S, K, T, r, q, kappa, theta, sigma, lam, v0, rho, trap, n):
    x, w = GenerateGaussLaguerre(n)
    int1 = []
    int2 = []
    for k, xi in enumerate(x):
        int1.append(w[k] * HestonProb(xi, kappa, theta, lam, rho, sigma, T, K, S, r, q, v0, 1, trap))
        int2.append(w[k] * HestonProb(xi, kappa, theta, lam, rho, sigma, T, K, S, r, q, v0, 2, trap))

    P1 = 0.5 + (1.0 / np.pi) * np.sum(int1)
    P2 = 0.5 + (1.0 / np.pi) * np.sum(int2)

    HestonC = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    HestonP = HestonC - S * np.exp(-q * T) + K * np.exp(-r * T)

    if PutCall == 'C':
        y = HestonC
    else:
        y = HestonP

    return y
