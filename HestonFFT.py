import numpy as np

def HestonCF(phi, kappa, theta, lam, rho, sigma, tau, S, r, q, v0, Trap):
    x = np.log(S)
    a = kappa * theta
    u = -0.5
    b = kappa + lam

    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if Trap == 1:
        # Little Trap
        c = 1.0 / g
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi - d) * tau - 2 * np.log(G))
        D = (b - rho * sigma * 1j * phi - d) / sigma ** 2 * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
    elif Trap == 0:
        G = (1 - g * np.exp(d * tau)) / (1 - g)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * ((b - rho * sigma * 1j * phi + d) * tau - 2 * np.log(G))
        D = (b - rho * sigma * 1j * phi + d) / sigma ** 2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    else:
        raise ValueError("Trap parameter must be either 0 (Original Heston) or 1 (Little Heston Trap).")

    y = np.exp(C + D * v0 + 1j * phi * x)

    return y


def HestonCallFFT(N, uplimit, S0, r, q, tau, kappa, theta, lam, rho, sigma, v0, Trap, alpha, rule):
    s0 = np.log(S0)

    eta = uplimit / N
    lambdainc = (2 * np.pi) / (N * eta)

    w = np.ones(N)
    if rule == 'T':  # Trapezoidal rule
        w[0] = 0.5
        w[-1] = 0.5
    elif rule == 'S':  # Simpson's rule
        w[0] = 1.0 / 3.0
        w[-1] = 1.0 / 3.0
        for k in range(1, N - 1):
            if (k + 1) % 2 == 0:
                w[k] = 4.0 / 3.0
            else:
                w[k] = 2.0 / 3.0
    else:
        raise ValueError("rule parameter must be 'T' (Trapezoidal) or 'S' (Simpson)")

    b = N * lambdainc / 2.0
    v = eta * np.arange(N)  # v is an array of length N
    k_grid = -b + lambdainc * np.arange(N) + s0
    K = np.exp(k_grid)
    U = np.arange(N)
    J = np.arange(N)

    psi = HestonCF(v - (alpha + 1) * 1j, kappa, theta, lam, rho, sigma, tau, S0, r, q, v0, Trap)
    phi = np.exp(-r * tau) * psi / (alpha ** 2 + alpha - v ** 2 + 1j * v * (2 * alpha + 1))
    x = np.exp(1j * (b - s0) * v) * phi * w

    E = np.exp(-1j * 2 * np.pi / N * np.outer(U, J))
    e = E.dot(x)

    CallFFT = eta * np.exp(-alpha * k_grid) / np.pi * np.real(e)

    return CallFFT, K, lambdainc, eta




CallFFT, K, lambdainc, eta = HestonCallFFT(500, 1000, 100, 0.05,0.01, 1.5, 2, 0.05, 0, 0.45, 0.3, 0.05, 1, 1.5, 'S')