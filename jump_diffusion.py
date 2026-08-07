import numpy as np

def sim_jump_diffusion(S0, mu, sigma, lambda_, mu_jump, sigma_jump, years, runs, days_per_year=252):

    days = days_per_year * years
    dt = 1 / days_per_year

    # GBM randomness
    Z = np.random.normal(0, 1, (runs, days))

    gbm_part = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )

    # Number of jumps each day
    jumps = np.random.poisson(
        lambda_ * dt,
        (runs, days)
    )

    # Jump sizes
    jump_sizes = np.random.normal(
        mu_jump,
        sigma_jump,
        (runs, days)
    )

    jump_part = jumps * jump_sizes

    # Combine GBM + jumps
    exponent = gbm_part + jump_part

    prices = S0 * np.cumprod(
        np.exp(exponent),
        axis=1
    )

    return prices