import numpy as np
import math


def GenerateGaussLaguerre(n):
    """
    Generate abscissas (x) and weights (w) for Gauss-Laguerre integration.

    Parameters
    ----------
    n : int
        Number of abscissas and weights.

    Returns
    -------
    x : numpy.ndarray
        The abscissas (roots) for the Gauss-Laguerre integration.
    w : numpy.ndarray
        The corresponding weights.
    """

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
    """
    Returns the integrand for the risk neutral probabilities P1 and P2.

    Parameters:
    phi   : integration variable (can be a scalar or array)
    kappa : volatility mean reversion speed parameter
    theta : volatility mean reversion level parameter
    lam   : risk parameter (originally lambda in MATLAB)
    rho   : correlation between two Brownian motions
    sigma : volatility of variance
    tau   : time to maturity
    K     : strike price
    S     : spot price
    r     : risk free rate
    q     : dividend yield
    v     : initial variance
    Pnum  : 1 or 2 (for the probabilities)
    Trap  : 1 for "Little Trap" formulation, 0 for original Heston formulation

    Returns:
    y : real part of the integrand
    """

    # Log of the stock price.
    x = np.log(S)

    # Parameter "a" is the same for P1 and P2.
    a = kappa * theta

    # Parameters "u" and "b" differ for P1 and P2.
    if Pnum == 1:
        u = 0.5
        b = kappa + lam - rho * sigma
    else:
        u = -0.5
        b = kappa + lam

    d = np.sqrt((rho * sigma * 1j * phi - b) ** 2 - sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if Trap == 1:
        # "Little Heston Trap" formulation
        c = 1.0 / g
        D = ((b - rho * sigma * 1j * phi - d) / sigma ** 2) * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * (((b - rho * sigma * 1j * phi - d) * tau) - 2 * np.log(G))
    elif Trap == 0:
        # Original Heston formulation.
        G = (1 - g * np.exp(d * tau)) / (1 - g)
        C = (r - q) * 1j * phi * tau + a / sigma ** 2 * (((b - rho * sigma * 1j * phi + d) * tau) - 2 * np.log(G))
        D = ((b - rho * sigma * 1j * phi + d) / sigma ** 2) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    else:
        raise ValueError("Trap parameter must be either 1 (Little Trap) or 0 (Original Heston).")

    # The characteristic function.
    f = np.exp(C + D * v + 1j * phi * x)

    # Return the real part of the integrand.
    y = np.real(np.exp(-1j * phi * np.log(K)) * f / (1j * phi))

    return y


def HestonPriceGaussLaguerre(PutCall, S, K, T, r, q, kappa, theta, sigma, lam, v0, rho, trap, n):
    """
    Heston (1993) call or put price by Gauss-Laguerre Quadrature.

    Uses the original Heston formulation of the characteristic function,
    or the "Little Heston Trap" formulation.

    Parameters
    ----------
    PutCall : str
        'C' for Call or 'P' for Put.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk free rate.
    q : float
        Dividend yield.
    kappa : float
        Heston parameter: mean reversion speed.
    theta : float
        Heston parameter: mean reversion level.
    sigma : float
        Heston parameter: volatility of volatility.
    lam : float
        Heston parameter: risk (originally lambda in MATLAB, but lambda is reserved in Python).
    v0 : float
        Heston parameter: initial variance.
    rho : float
        Heston parameter: correlation.
    trap : int
        1 for "Little Trap" formulation, 0 for original Heston formulation.
    n: int
        Polynomial degree.

    Returns
    -------
    y : float
        The Heston call or put price.
    """
    # Numerical integration using Gauss-Laguerre quadrature.
    x, w = GenerateGaussLaguerre(n)
    int1 = []
    int2 = []
    for k, xi in enumerate(x):
        int1.append(w[k] * HestonProb(xi, kappa, theta, lam, rho, sigma, T, K, S, r, q, v0, 1, trap))
        int2.append(w[k] * HestonProb(xi, kappa, theta, lam, rho, sigma, T, K, S, r, q, v0, 2, trap))

    # Define P1 and P2.
    P1 = 0.5 + (1.0 / np.pi) * np.sum(int1)
    P2 = 0.5 + (1.0 / np.pi) * np.sum(int2)

    # The call price.
    HestonC = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    # The put price by put-call parity.
    HestonP = HestonC - S * np.exp(-q * T) + K * np.exp(-r * T)

    # Output the option price based on PutCall flag.
    if PutCall == 'C':
        y = HestonC
    else:
        y = HestonP

    return y
