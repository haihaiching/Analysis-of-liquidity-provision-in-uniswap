from web3 import Web3
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timezone
from decimal import Decimal
import tqdm
import ast
import json
import time
import os
import pickle

POOL_ABI = json.loads('''[
    {"anonymous":false,"inputs":[
        {"indexed":true,"name":"sender","type":"address"},
        {"indexed":true,"name":"recipient","type":"address"},
        {"indexed":false,"name":"amount0","type":"int256"},
        {"indexed":false,"name":"amount1","type":"int256"},
        {"indexed":false,"name":"sqrtPriceX96","type":"uint160"},
        {"indexed":false,"name":"liquidity","type":"uint128"},
        {"indexed":false,"name":"tick","type":"int24"}
    ],"name":"Swap","type":"event"},
    {"anonymous":false,"inputs":[
        {"indexed":false,"name":"sender","type":"address"},
        {"indexed":true,"name":"owner","type":"address"},
        {"indexed":true,"name":"tickLower","type":"int24"},
        {"indexed":true,"name":"tickUpper","type":"int24"},
        {"indexed":false,"name":"amount","type":"uint128"},
        {"indexed":false,"name":"amount0","type":"uint256"},
        {"indexed":false,"name":"amount1","type":"uint256"}
    ],"name":"Mint","type":"event"},
    {"anonymous":false,"inputs":[
        {"indexed":true,"name":"owner","type":"address"},
        {"indexed":true,"name":"tickLower","type":"int24"},
        {"indexed":true,"name":"tickUpper","type":"int24"},
        {"indexed":false,"name":"amount","type":"uint128"},
        {"indexed":false,"name":"amount0","type":"uint256"},
        {"indexed":false,"name":"amount1","type":"uint256"}
    ],"name":"Burn","type":"event"}
]''')

POOL_LIQUIDITY_ABI = json.loads('''[
    {"inputs":[],"name":"liquidity","outputs":[{"type":"uint128"}],"stateMutability":"view","type":"function"}
]''')

CHUNK_SIZE = 10
SLEEP      = 0.2
CACHE_DIR  = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TOPICS = {
    'Swap': '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67',
    'Mint': '0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde',
    'Burn': '0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c',
}


def _cache_path(pool_id, event_name, start_block, end_block):
    return os.path.join(CACHE_DIR, f"{pool_id[:8]}_{event_name}_{start_block}_{end_block}.pkl")


def _fetch_raw_logs(web3, pool_address, topic, event_name, start_block, end_block):
    cache_file = _cache_path(pool_address, event_name, start_block, end_block)

    # load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        last_block = cached['last_block']
        records    = cached['records']
        if last_block >= end_block:
            print(f"    [{event_name}] loaded from cache ({len(records)} logs)")
            return records
        cur = last_block + 1
        print(f"    [{event_name}] resuming from block {cur} ({len(records)} logs so far)")
    else:
        records    = []
        cur        = start_block

    while cur <= end_block:
        nxt = min(cur + CHUNK_SIZE - 1, end_block)
        try:
            logs = web3.eth.get_logs({
                'fromBlock': cur,
                'toBlock':   nxt,
                'address':   pool_address,
                'topics':    [topic],
            })
            records.extend(logs)
        except Exception as e:
            # save progress before raising
            with open(cache_file, 'wb') as f:
                pickle.dump({'last_block': cur - 1, 'records': records}, f)
            print(f"    [{event_name}] saved progress at block {cur - 1}, error: {e}")
            raise
        cur = nxt + 1
        time.sleep(SLEEP)

    # save completed cache
    with open(cache_file, 'wb') as f:
        pickle.dump({'last_block': end_block, 'records': records}, f)

    return records


def load_data_web3(start_block, end_block, CONFIG_FILE, RPC_ENDPOINT, label):
    with open(CONFIG_FILE, 'r') as f:
        conf = json.load(f)

    pool_id   = Web3.to_checksum_address(conf["pool_id"])
    token0    = conf["token0"]
    token1    = conf["token1"]
    decimal_0 = conf["decimal_0"]
    decimal_1 = conf["decimal_1"]
    fee_tier  = conf["fee_tier"]

    web3         = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    assert web3.is_connected(), "Cannot connect to RPC"
    contract     = web3.eth.contract(address=pool_id, abi=POOL_ABI)
    liq_contract = web3.eth.contract(address=pool_id, abi=POOL_LIQUIDITY_ABI)

    # ── fetch raw logs ─────────────────────────────────────────────────────
    print("  Fetching Swap...")
    swap_raw = _fetch_raw_logs(web3, pool_id, TOPICS['Swap'], 'Swap', start_block, end_block)
    print("  Fetching Mint...")
    mint_raw = _fetch_raw_logs(web3, pool_id, TOPICS['Mint'], 'Mint', start_block, end_block)
    print("  Fetching Burn...")
    burn_raw = _fetch_raw_logs(web3, pool_id, TOPICS['Burn'], 'Burn', start_block, end_block)

    # ── decode logs ────────────────────────────────────────────────────────
    swap_logs = [contract.events.Swap().process_log(r) for r in swap_raw]
    mint_logs = [contract.events.Mint().process_log(r) for r in mint_raw]
    burn_logs = [contract.events.Burn().process_log(r) for r in burn_raw]

    # ── extract timestamps from raw logs ──────────────────────────────────
    def get_ts(raw):
        return int(raw['blockTimestamp'], 16)

    swap_ts = {r['blockNumber']: get_ts(r) for r in swap_raw}
    mint_ts = {r['blockNumber']: get_ts(r) for r in mint_raw}
    burn_ts = {r['blockNumber']: get_ts(r) for r in burn_raw}

    # ── cache pool liquidity per block (for Mint/Burn) ─────────────────────
    mb_blocks = {log['blockNumber'] for log in mint_logs + burn_logs}
    print(f"  Fetching pool liquidity for {len(mb_blocks)} blocks...")
    liquidity_cache = {}
    for bn in sorted(mb_blocks):
        liquidity_cache[bn] = str(liq_contract.functions.liquidity().call(block_identifier=bn))
        time.sleep(SLEEP)

    # ── parse swaps ────────────────────────────────────────────────────────
    swap_rows = []
    for log, raw in zip(swap_logs, swap_raw):
        args  = log['args']
        bn    = log['blockNumber']
        sqrtP = Decimal(args['sqrtPriceX96'])
        price = float((sqrtP / Decimal(2**96))**2 * Decimal(10**(int(decimal_0) - int(decimal_1))))
        swap_rows.append({
            'blockNumber': bn,
            'logIndex':    log['logIndex'],
            'datetime':    datetime.fromtimestamp(swap_ts[bn], tz=timezone.utc),
            'id':          log['transactionHash'].hex(),
            'event':       'Swap',
            'price':       price,
            'tick':        args['tick'],
            'liquidity':   str(args['liquidity']),
            token0:        args['amount0'] / 10**int(decimal_0),
            token1:        args['amount1'] / 10**int(decimal_1),
        })

    # ── parse mints ────────────────────────────────────────────────────────
    mint_rows = []
    for log, raw in zip(mint_logs, mint_raw):
        args = log['args']
        bn   = log['blockNumber']
        mint_rows.append({
            'blockNumber': bn,
            'logIndex':    log['logIndex'],
            'datetime':    datetime.fromtimestamp(mint_ts[bn], tz=timezone.utc),
            'id':          log['transactionHash'].hex(),
            'event':       'Mint',
            'owner':       args['owner'],
            'tickLower':   args['tickLower'],
            'tickUpper':   args['tickUpper'],
            'amount':      str(args['amount']),
            'liquidity':   liquidity_cache[bn],
            token0:        args['amount0'] / 10**int(decimal_0),
            token1:        args['amount1'] / 10**int(decimal_1),
        })

    # ── parse burns ────────────────────────────────────────────────────────
    burn_rows = []
    for log, raw in zip(burn_logs, burn_raw):
        args = log['args']
        bn   = log['blockNumber']
        burn_rows.append({
            'blockNumber': bn,
            'logIndex':    log['logIndex'],
            'datetime':    datetime.fromtimestamp(burn_ts[bn], tz=timezone.utc),
            'id':          log['transactionHash'].hex(),
            'event':       'Burn',
            'owner':       args['owner'],
            'tickLower':   args['tickLower'],
            'tickUpper':   args['tickUpper'],
            'amount':      str(args['amount']),
            'liquidity':   liquidity_cache[bn],
            token0:        args['amount0'] / 10**int(decimal_0),
            token1:        args['amount1'] / 10**int(decimal_1),
        })

    # ── combine & sort ─────────────────────────────────────────────────────
    df = pd.concat([
        pd.DataFrame(swap_rows),
        pd.DataFrame(mint_rows),
        pd.DataFrame(burn_rows),
    ], ignore_index=True)
    df = df.sort_values(by=['blockNumber', 'logIndex'], ascending=True).reset_index(drop=True)

    fee_str  = str(round(float(fee_tier) * 100, 4)).rstrip('0').rstrip('.')
    out_path = f"data/{token0}_{token1}_{fee_str}_{label}.parquet"
    pl.from_pandas(df).write_parquet(out_path)
    print(f"  Saved {len(df):,} rows → {out_path}")

    return pl.from_pandas(df)

def patch_transaction_from(
    parquet_path: str,
    RPC_ENDPOINT: str,
    output_path: str = None,
    batch_size: int  = 100,
) -> pl.DataFrame:
    """
    Read existing parquet, fetch transactionFrom for all Mint/Burn rows,
    and write back to parquet. Supports resume on failure.
    """
    df = pl.read_parquet(parquet_path)
    output_path = output_path or parquet_path

    # only need tx hashes from Mint and Burn
    mb_df     = df.filter(pl.col("event").is_in(["Mint", "Burn"]))
    tx_hashes = mb_df["id"].unique().to_list()
    print(f"Unique Mint/Burn tx hashes: {len(tx_hashes)}")

    # load cache if exists
    cache_file = os.path.join(
        CACHE_DIR, f"patch_tx_from_{os.path.basename(parquet_path)}.pkl"
    )
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            tx_from_cache = pickle.load(f)
        print(f"Loaded {len(tx_from_cache)} cached tx senders")
    else:
        tx_from_cache = {}

    # fetch missing
    missing = [h for h in tx_hashes if h not in tx_from_cache]
    print(f"Fetching {len(missing)} missing tx senders...")

    web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    assert web3.is_connected(), "Cannot connect to RPC"

    for i, tx_hash in enumerate(tqdm.tqdm(missing)):
        try:
            tx_from_cache[tx_hash] = web3.eth.get_transaction(tx_hash)['from']
        except Exception as e:
            print(f"Failed to fetch tx {tx_hash}: {e}")
            tx_from_cache[tx_hash] = None

        # save cache every batch_size fetches
        if (i + 1) % batch_size == 0:
            with open(cache_file, 'wb') as f:
                pickle.dump(tx_from_cache, f)
            print(f"    checkpoint saved ({i + 1}/{len(missing)})")

        time.sleep(SLEEP)

    # final save
    with open(cache_file, 'wb') as f:
        pickle.dump(tx_from_cache, f)
    print(f"Saved cache to {cache_file}")

    # join back to df
    tx_from_series = df["id"].map_elements(
        lambda x: tx_from_cache.get(x, None),
        return_dtype=pl.String,
    )
    df = df.with_columns(tx_from_series.alias("transactionFrom"))

    df.write_parquet(output_path)
    return df

'''
def _post_with_retry(url, query, headers, retries=10):
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(url, json={"query": query}, headers=headers, timeout=120)
            return res
        except Exception as e:
            print(f"  [Retry {attempt}/{retries}] {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded due to SSLError")


def load_data(start_ts, end_ts, CONFIG_FILE, API_KEY, label):
    with open(CONFIG_FILE, 'r') as f:
        conf = json.load(f)

    pool_id    = conf["pool_id"].lower()
    token0     = conf["token0"]
    token1     = conf["token1"]
    decimal_0  = conf["decimal_0"]
    decimal_1  = conf["decimal_1"]
    fee_tier   = conf["fee_tier"]
    # uniswap base
    url        = "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # ── swap ──────────────────────────────────────────────────────────────
    swap_df = []
    last_ts = start_ts
    while True:
        query = f"""
        {{
        swaps(
        first:500,
        orderBy: timestamp,
        orderDirection: asc,
        where: {{
            pool: "{pool_id}"
            timestamp_lte: {end_ts}
            timestamp_gte: {last_ts}
        }})
        {{
        transaction {{ blockNumber }}
        timestamp
        id
        recipient
        sender
        logIndex
        amount0
        amount1
        amountUSD
        sqrtPriceX96
        tick
        }}
        }}"""
        res = _post_with_retry(url, query, headers)
        data = res.json()
        if 'errors' in data:
            print(f"  [swap query error] {data['errors']}")
            break
        _ = pd.DataFrame(res.json()['data']['swaps'])
        if len(_) == 0:
            break
        swap_df.append(_)
        new_ts = int(_['timestamp'].iloc[-1])
        if new_ts == last_ts:
            break
        last_ts = new_ts

    # ── mint ──────────────────────────────────────────────────────────────
    mint_df = []
    last_ts = start_ts
    while True:
        query = f"""
        {{
        mints(
        first:500,
        orderBy: timestamp,
        orderDirection: asc,
        where: {{
            pool: "{pool_id}"
            timestamp_lte: {end_ts}
            timestamp_gte: {last_ts}
        }})
        {{
        transaction {{ blockNumber }}
        timestamp
        id
        owner
        logIndex
        amount0
        amount1
        amount
        amountUSD
        tickLower
        tickUpper
        }}
        }}"""
        res = _post_with_retry(url, query, headers)
        _ = pd.DataFrame(res.json()['data']['mints'])
        if len(_) == 0:
            break
        mint_df.append(_)
        new_ts = int(_['timestamp'].iloc[-1])
        if new_ts == last_ts:
            break
        last_ts = new_ts

    # ── burn ──────────────────────────────────────────────────────────────
    burn_df = []
    last_ts = start_ts
    while True:
        query = f"""
        {{
        burns(
        first:500,
        orderBy: timestamp,
        orderDirection: asc,
        where: {{
            pool: "{pool_id}"
            timestamp_lte: {end_ts}
            timestamp_gte: {last_ts}
        }})
        {{
        transaction {{ blockNumber }}
        timestamp
        id
        owner
        logIndex
        amount0
        amount1
        amount
        amountUSD
        tickLower
        tickUpper
        }}
        }}"""
        res = _post_with_retry(url, query, headers)
        _ = pd.DataFrame(res.json()['data']['burns'])
        if len(_) == 0:
            break
        burn_df.append(_)
        new_ts = int(_['timestamp'].iloc[-1])
        if new_ts == last_ts:
            break
        last_ts = new_ts

    # ── combine & post-process ────────────────────────────────────────────
    # swaps
    df_swap = pd.concat(swap_df, ignore_index=True)
    df_swap = df_swap.drop_duplicates(subset=['id'])
    df_swap.loc[:, "timestamp"]    = pd.to_numeric(df_swap["timestamp"])
    df_swap.loc[:, 'datetime']     = df_swap['timestamp'].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc))
    df_swap.loc[:, 'blockNumber']  = df_swap['transaction'].apply(lambda x: int(x['blockNumber']))
    df_swap.loc[:, 'logIndex']     = pd.to_numeric(df_swap['logIndex'])
    df_swap.loc[:, token0]      = pd.to_numeric(df_swap["amount0"])
    df_swap.loc[:, token1]      = pd.to_numeric(df_swap["amount1"])
    df_swap.loc[:, "amountUSD"]    = pd.to_numeric(df_swap["amountUSD"])
    df_swap.loc[:, 'sqrtPriceX96'] = pd.to_numeric(df_swap['sqrtPriceX96'].apply(Decimal))
    df_swap.loc[:, 'price']        = (df_swap['sqrtPriceX96'] / 2**96)**2 * 10**(int(decimal_0) - int(decimal_1))
    df_swap.loc[:, 'tick']         = pd.to_numeric(df_swap['tick'])
    df_swap.loc[:, 'event']        = 'Swap'
    df_swap = df_swap[['blockNumber', 'logIndex', 'datetime', 'id', 'event',
                        'price', 'tick', token0, token1, 'amountUSD']].copy()

    # mints & burns
    df_mint = pd.concat(mint_df, ignore_index=True)
    df_mint.loc[:, "event"] = "Mint"
    df_burn = pd.concat(burn_df, ignore_index=True)
    df_burn.loc[:, "event"] = "Burn"
    df_mb = pd.concat([df_mint, df_burn], ignore_index=True)
    df_mb = df_mb.drop_duplicates(subset=['id'])
    df_mb.loc[:, "timestamp"]   = pd.to_numeric(df_mb["timestamp"])
    df_mb.loc[:, 'datetime']    = df_mb['timestamp'].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc))
    df_mb.loc[:, 'blockNumber'] = df_mb['transaction'].apply(lambda x: int(x['blockNumber']))
    df_mb.loc[:, 'logIndex']    = pd.to_numeric(df_mb['logIndex'])
    df_mb.loc[:, 'amount']      = df_mb['amount'].apply(Decimal)
    df_mb.loc[:, 'amountUSD']   = pd.to_numeric(df_mb['amountUSD'])
    df_mb.loc[:, token0]        = df_mb["amount0"].astype(float)
    df_mb.loc[:, token1]        = df_mb["amount1"].astype(float)
    df_mb.loc[:, 'tickLower']   = pd.to_numeric(df_mb['tickLower'])
    df_mb.loc[:, 'tickUpper']   = pd.to_numeric(df_mb['tickUpper'])
    df_mb = df_mb[['blockNumber', 'logIndex', 'datetime', 'id', 'event',
                   token0, token1, 'amount', 'amountUSD', 'tickLower', 'tickUpper']].copy()

    # combine all & sort by blockNumber then logIndex
    df = pd.concat([df_swap, df_mb], ignore_index=True)
    df = df.sort_values(by=['blockNumber', 'logIndex'], ascending=True).reset_index(drop=True).copy()

    fee_str = str(round(float(fee_tier) * 100, 4)).rstrip('0').rstrip('.')
    df.to_csv(f"data/{token0}_{token1}_{fee_str}_{label}.csv", index=False)

    return df

'''