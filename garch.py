import numpy as np


def sim_garch(
    S0,
    mu,
    omega,
    alpha,
    beta,
    returns,
    years,
    runs=10000,
    days_per_year=252
):

    days = years * days_per_year

    prices = np.zeros((runs, days))

    prices[:, 0] = S0

    # GARCH was fitted on returns * 100,
    # so use the same scale here
    scaled_returns = returns * 100

    # Initial variance in GARCH scale
    variance = scaled_returns.var()

    # Previous return in GARCH scale
    previous_return = scaled_returns.iloc[-1]

    for t in range(1, days):

        # GARCH(1,1):
        # sigma_t^2 = omega + alpha*r_(t-1)^2 + beta*sigma_(t-1)^2
        variance = (
            omega
            + alpha * previous_return**2
            + beta * variance
        )

        # Random shock
        z = np.random.normal(
            0,
            1,
            runs
        )

        # Convert volatility back to decimal return scale
        simulated_returns = (
            mu
            + (np.sqrt(variance) / 100) * z
        )

        # Convert log returns to prices
        prices[:, t] = (
            prices[:, t-1]
            * np.exp(simulated_returns)
        )

        # Update previous return in GARCH scale
        previous_return = (
            simulated_returns.mean() * 100
        )

    return prices