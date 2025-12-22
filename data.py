import numpy as np
import pandas as pd
import pathlib


def generate_time_range(n_bars, t_start='2025-01-01 06:00:00', t_freq='min'):
    """Wrapper function for pd.date_range with consistent argument names."""
    return pd.date_range(periods=n_bars, start=t_start, freq=t_freq)


def generate_random_prices(
        rng, n_bars, base_price=1.0, return_loc=0, return_scale=0.001,
        t_start='2025-01-01 06:00:00', t_freq='min'):
    """
    Generate synthetic price data with normally distributed returns.
    Prices are generated as p_0 = base_price, p_{t+1} = p_t * (1 + r_{t+1}),
    r_1, ..., r_n ~ N(return_loc, return_scale). 
    r_t values are clipped to [-0.999, +infinity).
    Returns: DataFrame ('t', 'price')
    """
    returns = rng.normal(size=n_bars, loc=return_loc, scale=return_scale)
    returns = np.clip(returns, -0.999, None)

    prices = base_price * np.cumprod(returns + 1)
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
    If side_probs is an iterable, P(-1) = side_probs[0], P(1) = side_probs[1], additional elements are ignored.
    In both cases P(0) = 1 - P(-1) - P(1).
    Returns: DataFrame ('t', 'signal')
    """
    if isinstance(side_probs, (tuple, list, np.ndarray)):
        p_neg, p_pos, *_ = side_probs
    else:
        p_neg = p_pos = side_probs
    assert 0 <= p_neg + p_pos <= 1, "Invalid signal probabilities"

    signals = rng.choice([-1, 0, 1], p=[p_neg, 1-p_neg-p_pos, p_pos], size=n_bars)
    times = generate_time_range(n_bars, t_start, t_freq)
    return pd.DataFrame({'t': times, 'signal': signals})


def load_hist_data(symbol='aapl.us', start_date='20210101', end_date='20250101', cache_dir='data'):
    """
    Download historical daily price data from stooq.pl and cache it locally.
    Stooq is a free public market data provider which does not require authentication.
    The dates are expected to be of format 'YYYYMMDD'.
    Returned prices correspond to daily closing prices.
    Returns: DataFrame ('t', 'price')
    """
    pathlib.Path(cache_dir).mkdir(exist_ok=True)
    data_path = f'{cache_dir}/{symbol}_{start_date}_{end_date}.csv'
    if pathlib.Path(data_path).exists():
        return pd.read_csv(data_path, parse_dates=['t'])

    url = f'https://stooq.pl/q/d/l/?s={symbol}&i=d&f={start_date}&t={end_date}'
    df = pd.read_csv(url, parse_dates=['Data'])
    if df.empty or 'Data' not in df.columns:
        raise ValueError(f'Failed to download data for symbol={symbol} in range {start_date}-{end_date}.')
    # Stooq returns column names in Polish. Only the closing price is retained.
    df = df.drop(['Otwarcie', 'Najwyzszy', 'Najnizszy', 'Wolumen'], axis=1).rename({'Data': 't', 'Zamkniecie': 'price'}, axis=1)
    df = df.sort_values('t').reset_index(drop=True)
    df.to_csv(data_path, index=False)
    return df
