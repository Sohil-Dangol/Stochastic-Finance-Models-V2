import numpy as np

def calculate_errors(simulated, actual):
    return np.abs((np.array(simulated) - np.array(actual)) / np.array(actual)) * 100

def  calculate_simulated_returns(final_prices, S0):
    return (final_prices - S0) / S0

def sharpe_ratio(returns,risk_free = 0.04):
    return (np.mean(returns)-risk_free) / np.std(returns)


def value_at_risk(final_prices, confidence=95):

    return np.percentile(
        final_prices,
        100-confidence
    )

def confidence_interval_check(final_prices, actual_price, lower_percentile=5, upper_percentile=95):

    lower = np.percentile(final_prices, lower_percentile)
    upper = np.percentile(final_prices, upper_percentile)

    inside = lower <= actual_price <= upper

    return inside, lower, upper

def cumulative_returns(returns):
    return (np.prod(1 + np.array(returns)) - 1) * 100

def apply_strategy(strategy_name, expected_return, probability_up, actual_return,VaR):

    if strategy_name == "Expected Return":

        if expected_return > 0.10:
            return actual_return, "BUY"
        else:
            return 0, "HOLD"

    elif strategy_name == "Probability":

        if probability_up > 0.70:
            return actual_return, "BUY"
        else:
            return 0, "HOLD"

    elif strategy_name == "Probability + VaR":

        if probability_up > 0.70 and VaR > -15:
            return actual_return, "BUY"
        else:
            return 0, "HOLD"

    else:
        raise ValueError("Unknown strategy")