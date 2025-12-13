import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_time_range(n_bars, start_time='2025-01-01 06:00:00', freq='min'):
    return pd.date_range(start_time, periods=n_bars, freq=freq)


def generate_random_prices(rng, n_bars, base_price=1.0, return_loc=0, return_scale=0.001):
    returns = rng.normal(size=n_bars, loc=return_loc, scale=return_scale)
    prices = base_price * np.cumprod(returns + 1)  # assumes returns > -1
    # TODO: add arguments for time range adjustment
    times = generate_time_range(n_bars)
    return pd.DataFrame({'t': times, 'price': prices})


def generate_random_spreads(rng, n_bars, base_spread=0.0002, loc=0, scale=2):
    spreads = base_spread * np.exp(rng.normal(loc=loc, scale=scale, size=n_bars))
    # TODO: add arguments for time range adjustment
    times = generate_time_range(n_bars)
    return pd.DataFrame({'t': times, 'quoted_spread': spreads})


def generate_random_signal(rng, n_bars, side_probs=0.1):
    p_neg, p_pos = side_probs if np.ndim(side_probs) > 0 else (side_probs, side_probs)
    signals = rng.choice([-1, 0, 1], p=[p_neg, 1-p_neg-p_pos, p_pos], size=n_bars)
    # TODO: add arguments for time range adjustment
    times = generate_time_range(n_bars)
    return pd.DataFrame({'t': times, 'signal': signals})


def draw_results(return_df, price_df):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(return_df['t'], return_df['equity'])
    ymin, ymax = ax.get_ylim()
    ax.fill_between(return_df['t'], ymin, ymax, where=(return_df['r_position'] > 0), color='green', alpha=0.15)
    ax.fill_between(return_df['t'], ymin, ymax, where=(return_df['r_position'] < 0), color='red', alpha=0.15)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Equity')
    
    ax2 = ax.twinx()
    ax2.plot(price_df['t'], price_df['price'], c='black', alpha=0.3)
    ax2.set_ylabel('Price')
    return fig, ax, ax2