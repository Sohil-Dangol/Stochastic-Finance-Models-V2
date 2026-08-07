import numpy as np

def sim_gbm(S0,mu,sigma,years,runs,days_per_year=252):
    
    days = days_per_year * years
    dt = 1 / days_per_year
    Z = np.random.normal(0,1,(runs,days))
    
    exponent = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    
    returns = np.exp(exponent)
    
    prices = S0 * np.cumprod(
        returns,
        axis=1
    )

    return prices
    