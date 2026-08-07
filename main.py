import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model
from data_loader import *
from gbm import *
from Historical_Bootstrap import *
from jump_diffusion import *
from risk import *
from plots import *
from garch import *

# ----------------------------
# Configuration
# ----------------------------

TICKER = "AAPL"
START = "2001-01-01"
END = "2025-01-01"

TRAINING_YEARS = 10
SIM_YEARS = 1
RUNS = 10000

START_YEAR = 2005
LAST_YEAR = 2025

# ----------------------------
# Load Data
# ----------------------------

prices = load_stock(TICKER, START, END)["Close"].squeeze()


# ----------------------------
# Model Wrapper
# ----------------------------

def run_model(model_name, S0, returns, mu, sigma):

    if model_name == "GBM":

        return sim_gbm(
            S0,
            mu,
            sigma,
            years=SIM_YEARS,
            runs=RUNS
        )

    elif model_name == "Historical Bootstrap":

        return hist_bootstrap_sim(
            S0,
            returns,
            years=SIM_YEARS,
            runs=RUNS
        )

    elif model_name == "Jump Diffusion":

        threshold = returns.abs().quantile(0.99)

        jumps = returns[np.abs(returns) > threshold]

        lambda_ = len(jumps) / len(returns) * 252

        mu_jump = jumps.mean()

        sigma_jump = jumps.std()

        return sim_jump_diffusion(
            S0,
            mu,
            sigma,
            lambda_,
            mu_jump,
            sigma_jump,
            years=SIM_YEARS,
            runs=RUNS
        )
        
    elif model_name == "GARCH":
        
        mu = returns.mean()
        # Scale returns for arch package
        garch_returns = returns * 100

        model = arch_model(
            garch_returns,
            vol="GARCH",
            p=1,
            q=1
        )

        result = model.fit(
            disp="off"
        )

        omega = result.params["omega"]
        alpha = result.params["alpha[1]"]
        beta = result.params["beta[1]"]

        return sim_garch(
            S0,
            mu,
            omega,
            alpha,
            beta,
            returns,
            years=SIM_YEARS,
            runs=RUNS
        ) 
    else:
        raise ValueError("Unknown model")


# ----------------------------
# Evaluation
# ----------------------------


def evaluate_model(model_name,strategy):

    simulated_prices = []
    actual_prices = []

    confidence_results = []
    direction_results = []

    buy_hold_returns = []
    strategy_returns = []

    var_results = []
    es_results = []

    yearly_results = []


    for year in range(START_YEAR, LAST_YEAR - TRAINING_YEARS):

        train_start = f"{year}-01-01"
        train_end = f"{year+TRAINING_YEARS}-01-01"

        test_end = f"{year+TRAINING_YEARS+1}-01-01"


        train = prices.loc[train_start:train_end]
        test = prices.loc[train_end:test_end]


        returns = calculate_daily_returns(train)

        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)

        S0 = train.iloc[-1]


        simulation = run_model(
            model_name,
            S0,
            returns,
            mu,
            sigma
        )


        final_prices = simulation[:, -1]


        # -------------------------
        # Prediction
        # -------------------------

        predicted = np.mean(final_prices)

        actual = test.iloc[-1]


        simulated_prices.append(predicted)
        actual_prices.append(actual)

        # -------------------------
        # Risk metrics
        # -------------------------

        var_price = np.percentile(
            final_prices,
            5
        )


        VaR = (
            (var_price - S0)
            / S0
            * 100
        )


        worst_cases = final_prices[
            final_prices <= var_price
        ]


        es = (
            np.mean(worst_cases)-S0
        ) / S0 * 100


        var_results.append(VaR)
        es_results.append(es)
        
        # -------------------------
        # Probability strategy
        # -------------------------
        
        
        probability_up = np.mean(
            final_prices > S0
        )


        actual_return = (
            actual - S0
        ) / S0


        buy_hold_returns.append(
            actual_return
        )

        expected_return = (np.mean(final_prices) - S0) / S0

        strategy_return, decision = apply_strategy(
            strategy,
            expected_return,
            probability_up,
            actual_return,
            VaR
        )

        # -------------------------
        # Direction accuracy
        # -------------------------

        predicted_direction = probability_up > 0.5

        actual_direction = actual > S0

        direction_results.append(
            predicted_direction == actual_direction
        )

        # -------------------------
        # Confidence
        # -------------------------

        inside, lower, upper = confidence_interval_check(
            final_prices,
            actual
        )

        confidence_results.append(
            inside
        )


        yearly_results.append({

            "Year": year + TRAINING_YEARS + 1,
            "Probability Up": probability_up,
            "Decision": decision,
            "Predicted": predicted,
            "Actual": actual,
            "Actual Return (%)": actual_return*100,
            "Strategy Return (%)": strategy_return*100

        })


    # -------------------------
    # Final Metrics
    # -------------------------

    errors = calculate_errors(
        simulated_prices,
        actual_prices
    )


    summary = {

        "Model": model_name,

        "MAPE": np.mean(errors),

        "Std Error": np.std(errors),

        "Coverage (%)":
            np.mean(confidence_results)*100,

        "Directional Accuracy (%)":
            np.mean(direction_results)*100,

        "Value at Risk (%)":
            np.mean(var_results),

        "Expected Shortfall (%)":
            np.mean(es_results),

        "Buy and Hold Return (%)":
            cumulative_returns(
                buy_hold_returns
            ),

        "Strategy Return (%)":
            cumulative_returns(
                strategy_returns
            )

    }


    return (
        pd.DataFrame(yearly_results),
        summary
    )

# ----------------------------
# Run All Models
# ----------------------------

models = [

    "GBM",
    "Historical Bootstrap",
    "Jump Diffusion",
    "GARCH"

]

strategies = [
    "Probability",
    "Expected Return",
    "Probability + VaR"
]

comparison = []

for model in models:

    for strategy in strategies:

        print(
            f"Running {model} with {strategy}"
        )

        yearly, summary = evaluate_model(
            model,
            strategy
        )

        comparison.append(summary)



comparison = pd.DataFrame(comparison)

print("\n==============================")
print("Overall Comparison")
print("==============================")
pd.set_option("display.max_columns", None)
print(comparison)

comparison.to_csv("comparison.csv", index=False)