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
    return_df: DataFrame ('t', 'r_position', 'trade', 'net_return', 'equity')
    trade_df DataFrame ('t', 'price', 'trade')
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
    
    return_df = merged_df[['t', 'r_position', 'trade', 'net_return', 'equity']]
    
    trade_df = merged_df.loc[merged_df['trade'].abs() > 0, ['t', 'price', 'trade']]
    return return_df, trade_df