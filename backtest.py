import numpy as np
import pandas as pd


def backtester(price_df, signal_df):
    """
    Backtester prototype
    Execution strategy: on the first non-zero signal open position until signal sign switch. On signal sign switch open the inverse position instead.
    price_df: ('t', 'price')
    signal_df: ('t', 'signal')
    """
    # Merge prices with signals under condition that the signal timestamp is strictly lower than price timestamp.
    merged_df = pd.merge_asof(price_df, signal_df, on='t', allow_exact_matches=False, direction='backward')
    merged_df['signal'].fillna(0, inplace=True)
    merged_df['position'] = np.sign(merged_df['signal']).replace(0, np.nan).ffill().fillna(0)
    merged_df['trade'] = merged_df['position'].diff(1).fillna(0)
    merged_df['return'] = (merged_df['position'].shift(1) * (merged_df['price'].diff(1) / merged_df['price'].shift(1))).fillna(0)
    merged_df['equity'] = (1 + merged_df['return']).cumprod()
    # When analyzing return/equity values use positions that created those returns, thus the ones from the prev. bar
    merged_df['r_position'] = merged_df['position'].shift(1).fillna(0)
    
    return_df = merged_df[['t', 'r_position', 'return', 'equity']]
    
    trade_df = merged_df.loc[merged_df['trade'].abs() > 0, ['t', 'price', 'trade']]
    return return_df, trade_df