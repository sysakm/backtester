# Scalar Signal Backtester (Single-Asset, Fixed-Size)

Minimal vectorized backtester for single-asset strategies with a fixed position size, driven by an exogenous scalar trading signal.
Intended for educational use, experimentation, and as a minimal reference implementation.

## Overview

- Trades a single asset with position size in {−1, 0, 1}.
- Opens long position on positive signal.
- Opens short position on negative signal if shorts are enabled.
- Positions are held until the signal switches sign.
- Open positions are marked to market on bar close.

## Execution Model

- Signals are assumed to be generated externally.
- Trades are executed on the first bar **strictly after** the signal timestamp.
- With daily data, this corresponds to next-day closing price execution.
- In-bar microstructure is ignored.
- Trades are executed at closing prices with spread costs applied at execution.

## Scope and Non-Goals

This project is **not** intended to be a full-featured trading system or production-grade backtesting framework.
The scope of the project is intentionally minimal for reproducibility and readability. 

The following are explicitly out of scope:
- multi-asset backtesting,
- variable position sizing,
- pyramiding or partial position closing,
- multiple simultaneous positions,
- intrabar execution or order book modeling.

## Demonstration Notebooks

### 1. Backtester Settings Demo
`backtester_settings_demo.ipynb`

- Uses synthetic price data.
- Demonstrates:
  - different spread assumptions,
  - long-only vs long–short behavior.
- Synthetic data is generated with fixed random seeds for reproducibility.

### 2. Real Data Example
`historical_data_demo.ipynb`

- Applies a simple trading signal (difference between current price and its moving average) to historical market data (AAPL stock, 2021.01-2024.12).
- Compares backtesting results for long-only and long-short approaches to execution.
- Demonstrates resulting equity curves and provides a simple statistics report.
- Results are shown purely as a demonstration of backtester functionality.
  
Historical prices are used for illustrative purposes only and are downloaded from **Stooq** (stooq.pl), a public market data mirror, which provides free and unauthenticated data access.
