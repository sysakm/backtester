import numpy as np
import pandas as pd


def backtester(price_df, signal_df, spread=0., spread_is_relative=True, allow_shorts=True):
    """
    Backtester prototype
    Execution strategy and PnL calculation:
    the signal is considered exogenous, any trade is executed on the first bar strictly after the respective signal timestamp;
    signal > 0 - close the short position if there is one, open long if not already open;
    signal < 0 - close the long position if there is one, open short if not already open and if allow_shorts=True;
    signal = 0 - keep the position unchanged;
    absolute long and short sizes are kept equal to 1;
    mark-to-market returns are calculated using the position held over the bar ('r_position');
    spread costs are charged at execution time based on position changes ('trade') on the current bar.
    Arguments:
    price_df: DataFrame ('t', 'price')
    signal_df: DataFrame ('t', 'signal')
    spread: float OR DataFrame ('t', 'quoted_spread'), default 0.
    spread_is_relative: bool, default True, if False divide spread values by respective price
    allow_shorts: bool, default True, if False the strategy does not open short positions, longs are closed to zero on negative signal instead
    Returns:
    return_df: DataFrame ('t', 'price', 'quoted_spread', 'r_position', 'trade', 'net_return', 'equity')
    """
    # Merge prices with signals under condition that the signal timestamp is strictly lower than price timestamp.
    merged_df = pd.merge_asof(price_df, signal_df, on='t', allow_exact_matches=False, direction='backward')
    merged_df['signal'].fillna(0, inplace=True)
    # Merge in spread data
    if np.ndim(spread) > 0:
        merged_df = pd.merge_asof(merged_df, spread, on='t', allow_exact_matches=True, direction='backward')
        merged_df['quoted_spread'].fillna(0, inplace=True)
    else:
        merged_df['quoted_spread'] = spread
    if not spread_is_relative:
        merged_df['quoted_spread'] = merged_df['quoted_spread'] / merged_df['price']

    merged_df['position'] = np.sign(merged_df['signal']).replace(0, np.nan).ffill().fillna(0)
    if not allow_shorts:
        merged_df['position'] = np.where(merged_df['position'] > 0, merged_df['position'], 0)
    merged_df['trade'] = merged_df['position'].diff(1).fillna(0)
    # When analyzing return/equity values use positions that created those returns, thus the ones from the prev. bar
    merged_df['r_position'] = merged_df['position'].shift(1).fillna(0)
    merged_df['return'] = (merged_df['r_position'] * merged_df['price'].pct_change()).fillna(0)
    # Unlike mark-to-market returns, spread costs are calculated based on current-bar position update
    merged_df['spread_return'] = np.abs(merged_df['trade']) * merged_df['quoted_spread'] / 2

    merged_df['net_return'] = merged_df['return'] - merged_df['spread_return']
    merged_df['equity'] = (1 + merged_df['net_return']).cumprod()
    
    return_df = merged_df[['t', 'price', 'quoted_spread', 'r_position', 'trade', 'net_return', 'equity']].copy()
    return return_df


def build_trade_pairs(return_df):
    """
    Create opening/closing trade pairs based on DataFrame created by backtester.
    Assumes fixed position size in {-1, 0, 1}, no pyramiding or partial position closing.
    Arguments:
    return_df: DataFrame containing ('t', 'price', 'quoted_spread', 'trade')
    Returns:
    trade_df: DataFrame ('open_t', 'open_price', 'open_pos_change', 'close_t', 'close_price')
    """
    trade_df = return_df.loc[~np.isclose(return_df['trade'].abs(), 0), ['t', 'price', 'quoted_spread', 'trade']].copy()
    trade_df['price'] = trade_df['price'] * (1 + np.where(trade_df['trade'] > 0, 1, -1) * trade_df['quoted_spread'] / 2)
    trade_df.drop('quoted_spread', axis=1, inplace=True)
    # Any trade of absolute size 2 involves one position closing and one position opening
    size_one_trades = trade_df[np.isclose(trade_df['trade'].abs(), 1)].copy()
    size_two_trades = trade_df[np.isclose(trade_df['trade'].abs(), 2)].copy()
    size_two_trades['trade'] = size_two_trades['trade'] / 2
    trade_df = pd.concat([size_one_trades, size_two_trades, size_two_trades]).sort_values('t').reset_index(drop=True)
    assert np.isclose(trade_df['trade'].abs(), 1).all(), 'Unexpected trade size'
    # Under position assumptions the trades are alternating between opening and closing
    opening_trade_df = trade_df[::2].rename({'t': 'open_t', 'price': 'open_price', 'trade': 'open_pos_change'}, axis=1).reset_index(drop=True)
    closing_trade_df = trade_df[1::2].drop('trade', axis=1).rename({'t': 'close_t', 'price': 'close_price'}, axis=1).reset_index(drop=True)
    trade_df = pd.concat([opening_trade_df, closing_trade_df], axis=1)
    return trade_df


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
    