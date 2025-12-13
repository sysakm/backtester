import numpy as np
import pandas as pd


def backtester(price_df, signal_df, spread=0, spread_is_relative=True):
    """
    Backtester prototype
    Execution strategy: on the first non-zero signal open position until signal sign switch. On signal sign switch open the inverse position instead.
    price_df: DataFrame ('t', 'price')
    signal_df: DataFrame ('t', 'signal')
    spread: scalar value for constant spread OR DataFrame ('t', 'quoted_spread')
    spread_is_relative: bool flag, default True, if False divide spread values by respective price
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
    merged_df['trade'] = merged_df['position'].diff(1).fillna(0)
    # When analyzing return/equity values use positions that created those returns, thus the ones from the prev. bar
    merged_df['r_position'] = merged_df['position'].shift(1).fillna(0)
    
    merged_df['return'] = (merged_df['r_position'] * merged_df['price'].pct_change()).fillna(0)
    merged_df['spread_return'] = np.abs(merged_df['trade']) * merged_df['quoted_spread'] / 2
    merged_df['net_return'] = merged_df['return'] - merged_df['spread_return']
    merged_df['equity'] = (1 + merged_df['net_return']).cumprod()
    
    return_df = merged_df[['t', 'r_position', 'trade', 'net_return', 'equity']]
    
    trade_df = merged_df.loc[merged_df['trade'].abs() > 0, ['t', 'price', 'trade']]
    return return_df, trade_df