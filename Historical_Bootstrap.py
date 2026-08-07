import numpy as np

def hist_bootstrap_sim(S0, returns, years, runs=10000):
    days = int(252 * years)

    # Randomly sample historical returns with replacement
    sampled_returns = np.random.choice(returns, size=(runs, days), replace=True)

    # Compound the returns into price paths
    prices = S0 * np.cumprod(1 + sampled_returns, axis=1)

    return prices