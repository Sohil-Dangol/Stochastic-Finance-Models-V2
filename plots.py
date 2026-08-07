import matplotlib.pyplot as plt
import numpy as np

def plot_sim(prices):
    for i in range(20):
        plt.plot(prices[i])
        
    plt.title("GBM Simulations")
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.show()
    
def plot_hist(final_prices):
    plt.hist(
        final_prices,
        bins=50
    )

    plt.xlabel("Final Stock Price")
    plt.ylabel("Frequency")
    plt.title("Distribution of Simulated Final Prices")

    plt.show()
    
def plot_errors(errors):
    
    plt.bar(range(len(errors)), errors)

    plt.xlabel("Prediction Period")
    plt.ylabel("Error (%)")
    plt.title("GBM Prediction Error")
    plt.axhline(0, linestyle="--")

    plt.show()
     
    