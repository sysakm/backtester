# Scalar Signal Backtester (Single-Asset, Fixed-Size)

Models strategies trading a single asset with fixed position size.
Takes a scalar signal as an input, opens long positions when positive signal value is encountered, and, if shorts are allowed, opens short positions on negative signal.
Any open position is held unchanged until the signal switches sign.
PnL for open positions is marked to market on bar close.

Any signal is assumed to be generated externally.
Resulting trades are executed on the first bar _strictly after_ the signal timestamp.
With daily data, this corresponds to next-day closing price execution.
This approach is used as a simplified, reproducible execution model.
In-bar microstructure is ignored, trades are executed at closing prices with spread costs deducted at execution.

Two demonstration notebooks are provided.
The first notebook ('backtester_settings_demo.ipynb') works with synthetic price data and demonstrates the available backtesting settings: spread assumptions and whether the strategy is long-only or allows short positions.
Synthetical data is generated randomly with fixed seeding for experiment reproducibility.

The second notebook (not committed yet) provides an example using real market data.
It evaluates a simple trading signal on historical prices and provides the resulting statistics as a demonstration of backtester functionality.
Historical data is downloaded from Stooq (stooq.pl), a public market data mirror that provides free, unauthenticated CSV access.
Data acquired from this source is used for illustrative purposes only and was chosen for being reproducible and quick to set up without requiring API keys.
