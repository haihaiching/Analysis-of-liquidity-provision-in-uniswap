import polars as pl
import numpy as np
from joblib import Parallel, delayed
import json
from pathlib import Path

# ============================================================
# Utilities
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)

def _masked_mean(X: np.ndarray, active: np.ndarray) -> np.ndarray:
    num = (X * active).sum(axis=0)
    den = active.sum(axis=0)
    return np.where(den > 0, num / den, np.nan)

def _masked_std(X: np.ndarray, active: np.ndarray) -> np.ndarray:
    mu = _masked_mean(X, active)
    diff = X - mu[np.newaxis, :]
    sq = (diff ** 2 * active).sum(axis=0)
    den = active.sum(axis=0)
    return np.where(den > 0, np.sqrt(sq / den), np.nan)

# ============================================================
# Step 1: Market Context (Blocks DF)
# ============================================================

def build_blocks_df(df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, float]:
    aggregated = (
        df.filter(pl.col("event") == "Swap")
        .group_by("blockNumber")
        .agg([
            pl.col("tick").last().alias("tick_end"),
            pl.col("tick").first().alias("tick_start"),
            pl.len().alias("n_trades"),
            pl.col("USDC").abs().sum().alias("volume"),
            pl.col("USDC").sum().alias("net_volume"), 
        ])
    )

    blocks_df = (
        aggregated
        .with_columns([
            (pl.col("tick_end") - pl.col("tick_start")).alias("dp")
        ])
        .drop("tick_start")
        .sort("blockNumber")
    )

    all_blocks = blocks_df["blockNumber"].to_numpy()
    abv = float(blocks_df["volume"].mean())

    return blocks_df, all_blocks, abv

# ============================================================
# Step 2: Position Lifecycle (Positions DF)
# ============================================================

def _build_burn_queues(burns_pd):
    queues = {}
    for idx, row in burns_pd.iterrows():
        k = (row["agent"], row["tickLower"], row["tickUpper"])
        queues.setdefault(k, []).append((row["block_burn"], idx))
    for k in queues:
        queues[k].sort(key=lambda x: x[0])
    return queues

def build_positions_df(df: pl.DataFrame, last_block: int) -> pl.DataFrame:
    mints = (
        df.filter(pl.col("event") == "Mint")
        .select(["blockNumber", "transactionFrom", "tickLower", "tickUpper", "amount"])
        .rename({"blockNumber": "block_mint", "transactionFrom": "agent", "amount": "liquidity"})
        .with_columns(pl.col("liquidity").cast(pl.Float64))
        .sort("block_mint").to_pandas()
    )
    burns = (
        df.filter(pl.col("event") == "Burn")
        .select(["blockNumber", "transactionFrom", "tickLower", "tickUpper", "amount"])
        .rename({"blockNumber": "block_burn", "transactionFrom": "agent", "amount": "liq_burn"})
        .with_columns(pl.col("liq_burn").cast(pl.Float64))
        .sort("block_burn").to_pandas()
    )

    burn_queues = _build_burn_queues(burns)
    records = []

    for _, row in mints.iterrows():
        k = (row["agent"], row["tickLower"], row["tickUpper"])
        queue = burn_queues.get(k, [])
        matched_burn = None
        for i, (t_burn, _) in enumerate(queue):
            if t_burn >= row["block_mint"]:
                matched_burn = queue.pop(i)[0]
                break
        
        records.append({
            "agent": row["agent"],
            "tick_lower": row["tickLower"],
            "tick_upper": row["tickUpper"],
            "liquidity": row["liquidity"],
            "block_mint": int(row["block_mint"]),
            "block_burn": int(matched_burn) if matched_burn is not None else last_block,
        })
    return pl.DataFrame(records)

# ============================================================
# Step 4: Liquidity Projection Matrix
# ============================================================

def _agent_active_liq(alpha, grp, all_blocks, block_tick, block_index):
    result = {}
    for row in grp.iter_rows(named=True):
        k1, k2 = row["tick_lower"], row["tick_upper"]
        lx = row["liquidity"]
        t_mint, t_burn = row["block_mint"], row["block_burn"]

        active = all_blocks[(all_blocks >= t_mint) & (all_blocks <= t_burn)]
        for t in active:
            tick_t = block_tick.get(t)
            if tick_t is not None and k1 <= tick_t <= k2:
                bi = block_index[t]
                result[bi] = result.get(bi, 0.0) + lx
    return alpha, {bi: max(v, 0.0) for bi, v in result.items()}

def build_active_liq_matrix(positions_df, blocks_df, agents, all_blocks, n_jobs=-1):
    block_tick = dict(zip(blocks_df["blockNumber"].to_list(), blocks_df["tick_end"].to_list()))
    block_index = {b: i for i, b in enumerate(blocks_df["blockNumber"].to_list())}
    
    grouped = [(a, positions_df.filter(pl.col("agent") == a)) for a in agents]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_agent_active_liq)(a, g, all_blocks, block_tick, block_index) for a, g in grouped
    )

    M = np.zeros((len(blocks_df), len(agents)), dtype=np.float32)
    agent_idx_map = {a: i for i, a in enumerate(agents)}
    for alpha, block_liq in results:
        ai = agent_idx_map[alpha]
        for bi, liq in block_liq.items():
            M[bi, ai] = liq
    return M, agents

# ============================================================
# Step 5 & 6: Feature Computation
# ============================================================

def compute_matrix_features(M, blocks_df, agents, abv, fee_tier):
    volume = blocks_df["volume"].to_numpy()
    n_trades = blocks_df["n_trades"].to_numpy()
    dp = blocks_df["dp"].to_numpy()
    n_ag = len(agents)

    active = (M > 0).astype(np.float32)
    L_pool = M.sum(axis=1, keepdims=True)
    M_share = np.where(L_pool > 0, M / L_pool, 0.0)
    M_liq_norm = M / abv
    M_ntrades = np.outer(n_trades, np.ones(n_ag))
    M_dp = np.outer(dp, np.ones(n_ag))
    M_fee = np.outer(volume * fee_tier, np.ones(n_ag))

    return pl.DataFrame({
        "agent": agents,
        "n_active_blocks": active.sum(axis=0).tolist(),
        "mean_liq_share": _masked_mean(M_share, active).tolist(),
        "mean_active_liq": _masked_mean(M_liq_norm, active).tolist(),
        "sd_active_liq": _masked_std(M_liq_norm, active).tolist(),
        "mean_n_trades": _masked_mean(M_ntrades, active).tolist(),
        "sd_n_trades": _masked_std(M_ntrades, active).tolist(),
        "mean_dp": _masked_mean(M_dp, active).tolist(),
        "sd_dp": _masked_std(M_dp, active).tolist(),
        "mean_fees": _masked_mean(M_fee, active).tolist(),
        "sd_fees": _masked_std(M_fee, active).tolist(),
    })

def _agent_position_features(alpha, grp):
    n_pos = len(grp)
    if n_pos == 0:
        return {"agent": alpha, "n_positions": 0, "mean_width": np.nan, "sd_width": np.nan, "frac_jit": np.nan, "rebal_freq": np.nan}
    
    widths = (grp["tick_upper"] - grp["tick_lower"]).to_numpy().astype(float)
    grp_s = grp.sort("block_mint")
    mints, burns = grp_s["block_mint"].to_numpy(), grp_s["block_burn"].to_numpy()
    
    rebal_count = 0
    for i in range(n_pos):
        for j in range(n_pos):
            if i != j and burns[i] <= mints[j] <= (burns[i] + 5):
                rebal_count += 1
                break

    return {
        "agent": alpha, "n_positions": n_pos,
        "mean_width": float(widths.mean()),
        "sd_width": float(widths.std(ddof=0)) if n_pos > 1 else 0.0,
        "frac_jit": float((grp["block_mint"] == grp["block_burn"]).sum() / n_pos),
        "rebal_freq": float(rebal_count / n_pos),
    }

def compute_position_features(positions_df, agents, n_jobs=-1):
    grouped = [(a, positions_df.filter(pl.col("agent") == a)) for a in agents]
    results = Parallel(n_jobs=n_jobs)(delayed(_agent_position_features)(a, g) for a, g in grouped)
    return pl.DataFrame(results)

# ============================================================
# Entry Point
# ============================================================

COLUMN_ORDER = [
    "agent", "n_positions", "n_active_blocks", "mean_liq_share", "mean_active_liq", 
    "sd_active_liq", "mean_width", "sd_width", "rebal_freq", "frac_jit", 
    "mean_n_trades", "sd_n_trades", "mean_dp", "sd_dp", "mean_fees", "sd_fees"
]

def build_feature_matrix(parquet_path, config_path, rebal_window=5, n_jobs=-1, output_path="features_X.parquet"):
    config = load_config(config_path)
    fee_tier = config["fee_tier"]
    df = pl.read_parquet(parquet_path)

    blocks_df, all_blocks, abv = build_blocks_df(df)
    last_block = int(all_blocks.max())

    positions_df = build_positions_df(df, last_block)
    agents = positions_df["agent"].unique().sort().to_list()

    M, agents = build_active_liq_matrix(positions_df, blocks_df, agents, all_blocks, n_jobs)
    feat_matrix = compute_matrix_features(M, blocks_df, agents, abv, fee_tier)
    feat_pos = compute_position_features(positions_df, agents, n_jobs)

    X = feat_matrix.join(feat_pos, on="agent", how="left").select(COLUMN_ORDER)
    X.write_parquet(output_path)
    
    return X, {"blocks_df": blocks_df, "positions_df": positions_df, "M": M, "agents": agents}