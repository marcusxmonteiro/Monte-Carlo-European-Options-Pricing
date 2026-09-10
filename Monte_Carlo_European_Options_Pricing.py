import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(42)

# Parameters
S0 = 100 # Current stock price
K = 105 # Strike price
r = 0.05 # Risk-free interest rate
sigma = 0.2 # Volatility
T = 1 # Time to maturity in years
num_sims = 100000 # Number of simulated paths

def monte_carlo_option_prices(S0, K, r, sigma, T, num_sims):
    """
    Prices both a European call and put option using Monte Carlo simulation from GBM.
    Both prices are computed from the same simulated terminal stock prices (ST), since they share the same underlying random paths.
    Returns a tuple: (call_price, put_price).
    """
    Z = np.random.standard_normal(num_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    call_payoffs = np.maximum(ST - K, 0)
    put_payoffs = np.maximum(K - ST, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_price = np.exp(-r * T) * np.mean(put_payoffs)
    return call_price, put_price

# Single run demonstration
call_option_price, put_option_price = monte_carlo_option_prices(S0, K, r, sigma, T, num_sims)
print(f"Estimated European call option price: {call_option_price:.4f}")
print(f"Estimated European put option price: {put_option_price:.4f}")

# Black-Scholes formula for European call and put option price
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bs_call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
bs_put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

print(f"Black-Scholes call price: {bs_call_price:.4f}")
print(f"Monte Carlo call price: {call_option_price:.4f}")
print(f"Difference (call): {abs(bs_call_price - call_option_price):.4f}")
print(f"Black-Scholes put price: {bs_put_price:.4f}")
print(f"Monte Carlo put price: {put_option_price:.4f}")
print(f"Difference (put): {abs(bs_put_price - put_option_price):.4f}")

# Run simulation for each size - between 10^2 and 10^6 simulations, logarithmically spaced, with 20 points 
sim_sizes = np.logspace(2, 6, 20).astype(int) 
results = [monte_carlo_option_prices(S0, K, r, sigma, T, n) for n in sim_sizes] 
mc_call_prices = [res[0] for res in results]
mc_put_prices = [res[1] for res in results]

# Creates a figure with two subplots for call and put option convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left - call option
ax1.plot(sim_sizes, mc_call_prices, marker='o', color='blue', label='Monte Carlo Call')
ax1.axhline(y=bs_call_price, color='red', linestyle='--', label='Black-Scholes Call')
ax1.set_xscale('log')
ax1.set_xlabel('Number of simulations')
ax1.set_ylabel('Call Option Price')
ax1.set_title('Call Option Convergence')
ax1.legend()
ax1.grid(True, which="both", linestyle="--", alpha=0.5)

# Right - put option
ax2.plot(sim_sizes, mc_put_prices, marker='s', color='green', label='Monte Carlo Put')
ax2.axhline(y=bs_put_price, color='red', linestyle='--', label='Black-Scholes Put')
ax2.set_xscale('log')
ax2.set_xlabel('Number of simulations')
ax2.set_ylabel('Put Option Price')
ax2.set_title('Put Option Convergence')
ax2.legend()
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()