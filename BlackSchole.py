import numpy as np
from scipy.stats import norm


def BisecBSIV(PutCall, S, K, rf, q, T, a, b, MktPrice, Tol, MaxIter):
    BSC = lambda s, K, rf, q, v, T: (
            s * np.exp(-q * T) * norm.cdf((np.log(s / K) + (rf - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))) -
            K * np.exp(-rf * T) * norm.cdf(
        (np.log(s / K) + (rf - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T)) - v * np.sqrt(T))
    )

    BSP = lambda s, K, rf, q, v, T: (
            K * np.exp(-rf * T) * norm.cdf(
        -(np.log(s / K) + (rf - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T)) + v * np.sqrt(T)) -
            s * np.exp(-q * T) * norm.cdf(-(np.log(s / K) + (rf - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T)))
    )

    if PutCall == 'C':
        lowCdif = MktPrice - BSC(S, K, rf, q, a, T)
        highCdif = MktPrice - BSC(S, K, rf, q, b, T)
    else:
        lowCdif = MktPrice - BSP(S, K, rf, q, a, T)
        highCdif = MktPrice - BSP(S, K, rf, q, b, T)


    if lowCdif * highCdif > 0:
        return -1
    else:
        for _ in range(MaxIter):
            midP = (a + b) / 2.0
            if PutCall == 'C':
                midCdif = MktPrice - BSC(S, K, rf, q, midP, T)
            else:
                midCdif = MktPrice - BSP(S, K, rf, q, midP, T)
            if abs(midCdif) < Tol:
                break
            else:
                if midCdif > 0:
                    a = midP
                else:
                    b = midP
        return midP
