import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Return-based statistics ---
def sharpe_ratio(return_df, an_factor=252):
    """Calculate Sharpe Ratio annualized by `an_factor`. Assumes per-bar arithmetic returns."""
    mu = return_df['net_return'].mean()
    sigma = return_df['net_return'].std()
    sharpe = 0 if np.isclose(sigma, 0) else mu / sigma * np.sqrt(an_factor)
    return sharpe


def max_drawdown_magn(return_df):
    """Calculate maximum drawdown magnitude of equity curve."""
    return (return_df['equity'].cummax() - return_df['equity']).max()


def max_drawdown_duration(return_df):
    """Calculate maximum drawdown duration of equity curve, the result is measured in bars."""
    drawdown_mask = ~np.isclose(return_df['equity'].cummax() - return_df['equity'], 0)
    if not np.any(drawdown_mask):
        return 0  # no drawdowns, equity is monotonically rising
    # Add another zero to the end of the array to work around last timestamp being inside the drawdown
    drawdown_mask = np.array(list(drawdown_mask.astype(int)) + [0])
    drawdown_mask_diff = drawdown_mask[1:] - drawdown_mask[:-1]
    diff_index = np.arange(len(drawdown_mask_diff))[drawdown_mask_diff != 0]
    # Guaranteed that diff_index has even length, since we added 0 to the end earlier closing any open drawdown
    # The first value is always not in drawdown since max({x}) = x
    return np.max(diff_index[1::2] - diff_index[::2])


def number_of_trades(return_df):
    """Calculate number of individual trades."""
    return np.sum(~np.isclose(return_df['trade'], 0))


# --- Trade-based statistics ---
def number_of_held_positions(trade_df):
    """Calculate number of held positions, excluding the last unclosed position if there is one."""
    return len(trade_df.dropna())


def average_trade_pair_duration(trade_df, bar_length='1d'):
    """Calculate average time the position is held, the result is measured in bars."""
    durations = (trade_df['close_t'] - trade_df['open_t']).dropna()
    if durations.empty:
        return 0
    return durations.mean() / pd.Timedelta(bar_length)


def winrate(trade_df):
    """Calculate the percentage of trade pairs with positive PnL."""
    pl = (trade_df['close_price'] - trade_df['open_price']) * trade_df['open_pos_change']
    pl = pl.dropna()
    if pl.empty:
        return 0
    return np.mean(pl > 0)


# --- Statistics report ---
def generate_stats_report(return_df, trade_df, an_factor=252, bar_length='1d'):
    """Calculate return-based and trade-based statistics and combine them into one dict instance."""
    results = {}

    results['Annualized Sharpe ratio'] = sharpe_ratio(return_df, an_factor)
    results['Maximum drawdown magnitude, equity units'] = max_drawdown_magn(return_df)
    results['Maximum drawdown duration, bars'] = max_drawdown_duration(return_df)
    results['Number of trades'] = number_of_trades(return_df)

    results['Number of held positions'] = number_of_held_positions(trade_df)
    results['Average position holding period, bars'] = average_trade_pair_duration(trade_df, bar_length)
    results['Win rate (%)'] = np.round(winrate(trade_df), 3) * 100
    return results


def format_stats_for_display(stats_dct):
    """Convert float statistics into formatted string values."""
    format_dct = {
        'Annualized Sharpe ratio': '{:.3f}',
        'Maximum drawdown magnitude, equity units': '{:.3f}',
        'Maximum drawdown duration, bars': '{:.0f}',
        'Number of trades': '{:.0f}',
        'Number of held positions': '{:.0f}',
        'Average position holding period, bars': '{:.1f}',
        'Win rate (%)': '{:.0f}',
    }
    ret_dct = {}
    for stat in stats_dct:
        ret_dct[stat] = format_dct.get(stat, '{:.3f}').format(stats_dct[stat])
    return ret_dct


# --- Validation ---
def assert_pnl_invariant(return_df, trade_df):
    """
    Asserts bar-based and trade-based net PnL calculations give the same value.

    Note:
    This function computes PnL in price (currency) units, not percentage returns, for internal consistency check.
    The computation is not used for reporting strategy performance, which is evaluated using percentage-based PnL.
    """
    raw_pnl = (return_df['price'].diff(1) * return_df['r_position']).iloc[1:].sum()
    spread_pnl = (return_df['trade'].abs() * return_df['price'] * return_df['quoted_spread'] / 2).sum()
    pnl1 = raw_pnl - spread_pnl

    if trade_df.empty:  # if signal is constantly zero
        pnl2 = 0.0
    else:
        realized_pnl = ((trade_df['close_price'] - trade_df['open_price']) * trade_df['open_pos_change']).dropna().sum()
        pnl2 = realized_pnl

        last_price = return_df.iloc[-1]['price']  # mark the remaining open position to market
        last_trade = trade_df.iloc[-1]
        if pd.isna(last_trade['close_t']):
            pnl2 += last_trade['open_pos_change'] * (last_price - last_trade['open_price'])

    assert np.isclose(pnl1, pnl2), f'PnL mismatch, pnl1={pnl1}, pnl2={pnl2}'


# --- Visualization ---
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
