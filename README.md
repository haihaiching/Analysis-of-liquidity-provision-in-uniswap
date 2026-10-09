# Clustering Liquidity Provider Behavior on Uniswap V3

This project clusters liquidity provider (LP) events on a Uniswap V3 pool based on the pool's trading activity and liquidity state at the time each event occurs. The resulting clusters are interpreted as different LP behavior patterns, and the project then tests whether each cluster's net liquidity flow is related to price returns and LP profit and loss. The method is adapted from ClusterLOB (Zhang et al., 2025, [arXiv:2504.20349](https://arxiv.org/abs/2504.20349)), which clusters limit-order-book events, and is applied here to on-chain AMM liquidity.

## Data

The data is on-chain events from the Uniswap V3 WETH/USDC 0.05% pool on Arbitrum: liquidity additions and removals (`Mint`/`Burn`) and swaps (`Swap`). The raw data is not included in this repository.

## Repository Structure

| Path | Description |
|------|-------------|
| `data_loader.py` | Loads raw event data and converts it into a clean DataFrame. |
| `features.py` | Computes the features for each LP event and normalizes them. |
| `clustering.py` | Computes dependent variables, runs K-Means++ clustering and analyzes correlations between clusters and the dependent variables. |
| `performance.py` | Evaluates cluster performance (Sharpe ratio, PnL) and visualizes clusters (PCA, optimal K, UMAP/t-SNE). |
| `main.ipynb` | Runs the full workflow from start to finish. |
| `config/` | Pool configuration files (JSON). |
| `data/`, `data_feature/` | Raw and derived data. |

## Features

Each feature is computed for every `Mint`/`Burn` event.

| Feature | Meaning |
|---------|---------|
| `S_x` | The event's share of active liquidity when the position is in range. |
| `Delta_T_x` | Time since the price last moved into a different tick-spacing bucket. |
| `V_x` | Swap volume since that price move. |
| `D_L_x` | Distance from the current tick to the position's lower bound. |
| `D_U_x` | Distance from the current tick to the position's upper bound. |
| `L_x` | Value of the existing liquidity between the current price and the position's range. |

Each feature is normalized with a rolling z-score, and the result is stored as `<feature>_norm`. The clustering uses these normalized features.

## Dependent Variables

| Variable | Meaning |
|----------|---------|
| `CONR` | Log return of the pool price over the current hour. |
| `FRNB` | Log return of the pool price over the next hour. |
| `HPNL` | PnL per unit of liquidity (fees earned plus impermanent loss) for a position held until the end of the next hour. |
| `NLF` | Net liquidity flow of each cluster per hour (minted minus burned liquidity). |

## Setup

Requires Python 3.10+.

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn tqdm requests pyarrow umap-learn jupyter
```

## Usage

1. Choose a pool configuration in `config/` (the default is `WETH_USDC_arbitrum_config.json`).
2. Run the cells of `main.ipynb` in order.

Settings used in the notebook:

- Training period: 2024-09-01 to 2025-03-01
- Test period: 2025-03-01 to 2025-09-01
- Number of clusters: K = 3 (the model trained on the training set is reused on the test set)

Intermediate results are saved to `data_feature/`.

To analyze another pool, copy a JSON file in `config/` and change its fields to match the new pool.


## Results

Three clusters were identified and interpreted as Yield Chasing, Just-in-Time, and Range Order behavior.

| Cluster | Sharpe (train) | Sharpe (test) |
|---------|---------------:|--------------:|
| Yield Chasing | 0.19 | −0.59 |
| Just-in-Time | −0.01 | −0.92 |
| Range Order | 0.27 | −0.29 |
| No clustering (baseline) | 0.12 | −0.46 |

HPNL is measured over a one-hour horizon, while LP positions are often held much longer, so these results reflect short-term performance only.