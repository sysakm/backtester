import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_time_range(n_bars, start='2025-01-01 06:00:00', freq='min'):
    """Wrapper function for pd.date_range with consistent argument names"""
    return pd.date_range(periods=n_bars, start=start, freq=freq)


def generate_random_prices(
        rng, n_bars, base_price=1.0, return_loc=0, return_scale=0.001,
        t_start='2025-01-01 06:00:00', t_freq='min'):
    """
    Generate synthetic price data with normally distributed returns.
    Prices are generated as p_0 = base_price, p_{t+1} = p_t * (1 + r_{t+1}),
    r_1, ..., r_n ~ N(return_loc, return_scale).
    Returns: DataFrame ('t', 'price')
    """
    returns = rng.normal(size=n_bars, loc=return_loc, scale=return_scale)
    prices = base_price * np.cumprod(returns + 1)  # assumes returns > -1
    times = generate_time_range(n_bars, t_start, t_freq)
    return pd.DataFrame({'t': times, 'price': prices})


def generate_random_spreads(
        rng, n_bars, base_spread=0.0002, scale=2,
        t_start='2025-01-01 06:00:00', t_freq='min'):
    """
    Generate synthetic spread data as base spread value multiplied by lognormal random value.
    Spreads are generated as spr_t = base_spread * exp(x_t),
    x_1, ..., x_n ~ N(0, scale).
    Returns: DataFrame ('t', 'quoted_spread')
    """
    spreads = base_spread * np.exp(rng.normal(loc=0, scale=scale, size=n_bars))
    times = generate_time_range(n_bars, t_start, t_freq)
    return pd.DataFrame({'t': times, 'quoted_spread': spreads})


def generate_random_signal(rng, n_bars, side_probs=0.1, t_start='2025-01-01 06:00:00', t_freq='min'):
    """
    Generate synthetic signal example with each value from {-1, 0, 1}.
    If side_probs is float, P(-1) = P(1) = side_probs.
    If side_probs is 2-tuple, P(-1) = side_probs[0], P(1) = side_probs[1].
    In both cases P(0) = 1 - P(-1) - P(1).
    Returns: DataFrame ('t', 'signal')
    """
    p_neg, p_pos = side_probs if np.ndim(side_probs) > 0 else (side_probs, side_probs)
    signals = rng.choice([-1, 0, 1], p=[p_neg, 1-p_neg-p_pos, p_pos], size=n_bars)
    times = generate_time_range(n_bars, t_start, t_freq)
    return pd.DataFrame({'t': times, 'signal': signals})


def draw_results(ax, ax2, return_df, price_df, fig_title=None):
    """
    Draw equity curve and shade position sides into ax based on return_df.
    Draw background price curve into ax2 based on price_df.
    """
    ax.plot(return_df['t'], return_df['equity'])
    ymin, ymax = ax.get_ylim()
    ax.fill_between(return_df['t'], ymin, ymax, where=(return_df['r_position'] > 0), color='green', alpha=0.15)
    ax.fill_between(return_df['t'], ymin, ymax, where=(return_df['r_position'] < 0), color='red', alpha=0.15)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Equity')
    if fig_title is not None:
        ax.set_title(fig_title)
    
    ax2.plot(price_df['t'], price_df['price'], c='black', alpha=0.3)
    ax2.set_ylabel('Price')