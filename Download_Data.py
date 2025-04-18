import yfinance as yf
import pandas as pd
import random
import numpy as np
import os
from datetime import datetime

# ticker = yf.Ticker("SPY")
# options = ticker.option_chain(ticker.options[-1])
# calls = options.calls
#


np.random.seed(15)

r = 0.04013
q = 0.0128
ticker = yf.Ticker("SPY")
data = ticker.history(start="2024-12-01", end="2025-04-16")
S0 = data["Close"].iloc[-1]


res = []


today = pd.to_datetime('2025-04-15')

ticker = yf.Ticker("SPY")


for _ in [10, 15, 20]:#[5, 10, 15, 20, 25]:#[2, 11, 15, 19, 28]:
    date = pd.to_datetime(ticker.options[_])
    options = ticker.option_chain(ticker.options[_])
    calls = options.calls

    Tlist = []
    Klist = []
    IVlist = []
    MktPrice = []

    for item in calls.itertuples(index=True):
        if item.lastTradeDate.date() == today.date():
            Tlist.append((date - today).total_seconds() / (365 * 24 * 3600))
            Klist.append(item.strike)
            IVlist.append(item.impliedVolatility)
            MktPrice.append(item.lastPrice)

    idx = np.sort(random.sample(range(len(Tlist)), 30))
    Tlist = [Tlist[__] for __ in idx]
    Klist = [Klist[__] for __ in idx]
    IVlist = [IVlist[__] for __ in idx]
    MktPrice = [MktPrice[__] for __ in idx]


    res.append([Tlist, Klist, IVlist, MktPrice])

def returnData():
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = "saved_data"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"option_data_{timestamp}.npz")
    #
    # np.savez(save_path, res=res, r=r, q=q, S0=S0, allow_pickle=True)
    # print(f"Saved option data to: {save_path}")
    return res, r, q, S0


