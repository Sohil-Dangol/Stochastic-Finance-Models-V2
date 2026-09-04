import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model

from quant.data_loader import *
from quant.gbm import *
from quant.Historical_Bootstrap import *
from quant.jump_diffusion import *
from quant.risk import *
from plots import *
from quant.garch import *


# ============================================================
# Configuration
# ============================================================

tickers = [
    "AAPL",
    "TSLA",
    "NVDA",
    "KO",
    "SPY"
]

START = "2001-01-01"
END = "2025-01-01"

TRAINING_YEARS = 10
SIM_YEARS = 1
RUNS = 10000

START_YEAR = 2005
LAST_YEAR = 2025


# ============================================================
# Model Wrapper
# ============================================================

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

        jumps = returns[
            returns.abs() > threshold
        ]

        lambda_ = (
            len(jumps) / len(returns)
        ) * 252

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

        # arch works better with percentage returns
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

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    ticker,
    prices,
    model_name
):

    simulated_prices = []
    actual_prices = []

    confidence_results = []
    direction_results = []

    buy_hold_returns = []

    var_results = []
    es_results = []

    yearly_results = []


    # --------------------------------------------------------
    # Rolling yearly evaluation
    # --------------------------------------------------------

    for year in range(
        START_YEAR,
        LAST_YEAR - TRAINING_YEARS
    ):

        train_start = f"{year}-01-01"
        train_end = f"{year + TRAINING_YEARS}-01-01"
        test_end = f"{year + TRAINING_YEARS + 1}-01-01"


        # ----------------------------------------------------
        # Training / testing data
        # ----------------------------------------------------

        train = prices.loc[
            train_start:train_end
        ]

        test = prices.loc[
            train_end:test_end
        ]


        # ----------------------------------------------------
        # Returns
        # ----------------------------------------------------

        returns = calculate_daily_returns(
            train
        )

        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)

        S0 = train.iloc[-1]


        # ----------------------------------------------------
        # Monte Carlo simulation
        # ----------------------------------------------------

        simulation = run_model(
            model_name,
            S0,
            returns,
            mu,
            sigma
        )

        final_prices = simulation[:, -1]


        # ----------------------------------------------------
        # Store all final simulated prices
        # ----------------------------------------------------

        for run, price in enumerate(
            final_prices
        ):

            final_prices_all.append({

                "Ticker": ticker,

                "Model": model_name,

                "Year":
                    year + TRAINING_YEARS + 1,

                "Run": run,

                "Final Price": price

            })


        # ----------------------------------------------------
        # Store sample paths
        # ----------------------------------------------------

        sample_paths = simulation[:100]

        for run in range(
            sample_paths.shape[0]
        ):

            for day in range(
                sample_paths.shape[1]
            ):

                sample_paths_all.append({

                    "Ticker": ticker,

                    "Model": model_name,

                    "Year":
                        year + TRAINING_YEARS + 1,

                    "Run": run,

                    "Day": day + 1,

                    "Price":
                        sample_paths[run, day]

                })


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted = np.mean(
            final_prices
        )

        actual = test.iloc[-1]

        simulated_prices.append(
            predicted
        )

        actual_prices.append(
            actual
        )


        # ----------------------------------------------------
        # APE
        # ----------------------------------------------------

        ape = (
            abs(
                (actual - predicted)
                / actual
            )
        ) * 100


        # ----------------------------------------------------
        # VaR
        # ----------------------------------------------------

        var_price = np.percentile(
            final_prices,
            5
        )

        VaR = (
            (var_price - S0)
            / S0
        ) * 100


        # ----------------------------------------------------
        # Expected Shortfall
        # ----------------------------------------------------

        worst_cases = final_prices[
            final_prices <= var_price
        ]

        if len(worst_cases) > 0:

            es = (
                (np.mean(worst_cases) - S0)
                / S0
            ) * 100

        else:

            es = np.nan


        var_results.append(
            VaR
        )

        es_results.append(
            es
        )


        # ----------------------------------------------------
        # Probability Up
        # ----------------------------------------------------

        probability_up = np.mean(
            final_prices > S0
        )


        # ----------------------------------------------------
        # Actual return
        # ----------------------------------------------------

        actual_return = (
            (actual - S0)
            / S0
        )

        buy_hold_returns.append(
            actual_return
        )


        # ----------------------------------------------------
        # Directional accuracy
        # ----------------------------------------------------

        predicted_direction = (
            probability_up > 0.5
        )

        actual_direction = (
            actual > S0
        )

        direction_results.append(
            predicted_direction
            == actual_direction
        )


        # ----------------------------------------------------
        # Confidence interval
        # ----------------------------------------------------

        inside, lower, upper = (
            confidence_interval_check(
                final_prices,
                actual
            )
        )

        confidence_results.append(
            inside
        )


        # ----------------------------------------------------
        # Store model-level yearly results
        # ----------------------------------------------------

        yearly_results.append({

            "Ticker":
                ticker,

            "Model":
                model_name,

            "Year":
                year + TRAINING_YEARS + 1,

            "S0":
                S0,

            "Probability Up":
                probability_up,

            "Predicted":
                predicted,

            "Actual":
                actual,

            "Actual Return (%)":
                actual_return * 100,

            "APE (%)":
                ape,

            "VaR (%)":
                VaR,

            "ES (%)":
                es,

            "Lower CI":
                lower,

            "Upper CI":
                upper

        })


    # ========================================================
    # Final Model Metrics
    # ========================================================

    errors = calculate_errors(
        simulated_prices,
        actual_prices
    )


    summary = {

        "Ticker":
            ticker,

        "Model":
            model_name,

        "MAPE (%)":
            np.mean(errors),

        "Median APE (%)":
            np.median(errors),

        "Std Error":
            np.std(errors),

        "Coverage (%)":
            np.mean(
                confidence_results
            ) * 100,

        "Directional Accuracy (%)":
            np.mean(
                direction_results
            ) * 100,

        "Value at Risk (%)":
            np.mean(
                var_results
            ),

        "Expected Shortfall (%)":
            np.mean(
                es_results
            ),

        "Buy and Hold Return (%)":
            cumulative_returns(
                buy_hold_returns
            )

    }


    return (
        pd.DataFrame(yearly_results),
        summary
    )


# ============================================================
# Evaluate Strategy
# ============================================================

def evaluate_strategy(
    yearly_results,
    strategy
):

    strategy_returns = []

    results = []


    # --------------------------------------------------------
    # Each row = one model + one year
    # --------------------------------------------------------

    for _, row in yearly_results.iterrows():

        # Expected return predicted by model
        expected_return = (
            row["Predicted"]
            - row["S0"]
        ) / row["S0"]


        probability_up = (
            row["Probability Up"]
        )


        actual_return = (
            row["Actual Return (%)"]
            / 100
        )


        VaR = row["VaR (%)"]


        # ----------------------------------------------------
        # Apply strategy
        # ----------------------------------------------------

        strategy_return, decision = (
            apply_strategy(
                strategy,
                expected_return,
                probability_up,
                actual_return,
                VaR
            )
        )


        strategy_returns.append(
            strategy_return
        )


        # ----------------------------------------------------
        # Store strategy result
        # ----------------------------------------------------

        results.append({

            "Ticker":
                row["Ticker"],

            "Model":
                row["Model"],

            "Strategy":
                strategy,

            "Year":
                row["Year"],

            "Probability Up":
                probability_up,

            "Expected Return (%)":
                expected_return * 100,

            "Decision":
                decision,

            "Predicted":
                row["Predicted"],

            "Actual":
                row["Actual"],

            "Actual Return (%)":
                row["Actual Return (%)"],

            "VaR (%)":
                row["VaR (%)"],

            "ES (%)":
                row["ES (%)"],

            "Strategy Return (%)":
                strategy_return * 100

        })


    # --------------------------------------------------------
    # Cumulative strategy return
    # --------------------------------------------------------

    cumulative_strategy_return = (
        cumulative_returns(
            strategy_returns
        )
    )


    return (
        pd.DataFrame(results),
        cumulative_strategy_return
    )


# ============================================================
# Models and Strategies
# ============================================================

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


# ============================================================
# Global Result Lists
# ============================================================

comparison = []

yearly_results_all = []

strategy_results_all = []

final_prices_all = []

sample_paths_all = []


# ============================================================
# Run All Models
# ============================================================

for ticker in tickers:

    print(
        f"\n=============================="
    )

    print(
        f"Testing {ticker}"
    )

    print(
        f"=============================="
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    prices = load_stock(
        ticker,
        START,
        END
    )["Close"].squeeze()


    # ========================================================
    # Run each model ONCE
    # ========================================================

    for model in models:

        print(
            f"\nRunning {ticker} - {model}"
        )


        # ----------------------------------------------------
        # Model evaluation
        # ----------------------------------------------------

        yearly, summary = evaluate_model(
            ticker,
            prices,
            model
        )


        # ----------------------------------------------------
        # Store yearly model results
        # ----------------------------------------------------

        yearly_results_all.append(
            yearly
        )


        # ====================================================
        # Evaluate every strategy using SAME model results
        # ====================================================

        for strategy in strategies:

            print(
                f"    Strategy: {strategy}"
            )


            strategy_yearly, strategy_return = (
                evaluate_strategy(
                    yearly,
                    strategy
                )
            )


            # ------------------------------------------------
            # Store strategy results
            # ------------------------------------------------

            strategy_results_all.append(
                strategy_yearly
            )


            # ------------------------------------------------
            # Store comparison result
            # ------------------------------------------------

            comparison_row = summary.copy()

            comparison_row["Strategy"] = (
                strategy
            )

            comparison_row["Strategy Return (%)"] = (
                strategy_return
            )

            comparison.append(
                comparison_row
            )


# ============================================================
# Comparison DataFrame
# ============================================================

comparison = pd.DataFrame(
    comparison
)


# Put Ticker first
comparison = comparison[
    ["Ticker"]
    + [
        col
        for col in comparison.columns
        if col != "Ticker"
    ]
]


print(
    "\n=============================="
)

print(
    "Overall Comparison"
)

print(
    "=============================="
)


pd.set_option(
    "display.max_columns",
    None
)

print(
    comparison
)


comparison.to_csv(
    "comparison.csv",
    index=False
)


# ============================================================
# Yearly Model Results
# ============================================================

yearly_results_df = pd.concat(
    yearly_results_all,
    ignore_index=True
)


yearly_results_df = yearly_results_df.sort_values(
    ["Ticker", "Model", "Year"]
)


print(
    "\n=============================="
)

print(
    "Yearly Model Results"
)

print(
    "=============================="
)

print(
    yearly_results_df.to_string(
        index=False
    )
)


yearly_results_df.to_csv(
    "yearly_results.csv",
    index=False
)


# ============================================================
# Strategy Results
# ============================================================

strategy_results_df = pd.concat(
    strategy_results_all,
    ignore_index=True
)


strategy_results_df = strategy_results_df.sort_values(
    ["Ticker", "Model", "Strategy", "Year"]
)


print(
    "\n=============================="
)

print(
    "Strategy Results"
)

print(
    "=============================="
)

print(
    strategy_results_df.to_string(
        index=False
    )
)


strategy_results_df.to_csv(
    "strategy_results.csv",
    index=False
)


# ============================================================
# Final Price Distributions
# ============================================================

final_prices_df = pd.DataFrame(
    final_prices_all
)


final_prices_df = final_prices_df.sort_values(
    ["Ticker", "Model", "Year", "Run"]
)


final_prices_df.to_csv(
    "final_prices.csv",
    index=False
)


# ============================================================
# Sample Monte Carlo Paths
# ============================================================

sample_paths_df = pd.DataFrame(
    sample_paths_all
)


sample_paths_df = sample_paths_df.sort_values(
    ["Ticker", "Model", "Year", "Run", "Day"]
)


sample_paths_df.to_csv(
    "sample_paths.csv",
    index=False
)


print(
    "\nAll datasets saved successfully."
)
