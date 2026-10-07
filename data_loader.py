import pandas as pd
import requests
import json
import os
import glob
import numpy as np
from datetime import datetime, timezone

def tick_2_price(tick, decimal_0, decimal_1):
    return (1.0001 ** tick) * (10 ** (int(decimal_0) - int(decimal_1))) 

def transform_amount_0(amount, decimal_0):
    return amount / 10 ** int(decimal_0)

def transform_amount_1(amount, decimal_1):
    return amount / 10 ** int(decimal_1)

def query_event_data(pool_address, chain, start_timestamp, end_timestamp):
    """
    Queries event data from the Teahouse API.
    """
    url = f"https://swap-api.teahouse.finance/{chain}/pools/events/{pool_address}/{start_timestamp}/{end_timestamp}"
    try:
        # Timeout set to 30 seconds to prevent hanging
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API Request Error: {e}")
        return {"data": []}

def dict_2_dataframe(response):
    column_names = ['timestamp', 'blockNumber', 'event', 'amount0', 'amount1', 'amount', 'tick', 'liquidity', 'tickLower',"tickUpper"]
    merge_dict = {}
    for i in column_names:
        merge_dict[i] = []
    for i in response["data"]:
        for j in column_names:
            if j in merge_dict:
                value = i.get(j, np.nan)
                merge_dict[j].append(value)
    df = pd.DataFrame(merge_dict)
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df

def load_liquidity_data(pool_address, chain, start_timestamp, end_timestamp, save):
    df_list = []
    step = (end_timestamp-start_timestamp)//20
    for i in range(start_timestamp, end_timestamp+1, step):
        if i+step < end_timestamp:
            print(f"Getting liquidity data from {i} to {i+step}")
            response = query_event_data(pool_address,chain, i, i+step)
        else:
            print(f"Getting liquidity data from {i} to {end_timestamp}")
            response = query_event_data(pool_address,chain, i, end_timestamp)
        df = dict_2_dataframe(response)
        df_list.append(df)
    result_df = pd.concat(df_list, axis=0)
    if save:
        base_dir = os.getcwd()
        path = os.path.join(base_dir, 'data')
        if not os.path.exists(path):
            os.mkdir(path)
        start_str = datetime.fromtimestamp(start_timestamp).strftime('%Y%m%d')
        end_str = datetime.fromtimestamp(end_timestamp).strftime('%Y%m%d')
        filename = f"data_{start_str}_{end_str}.csv"
        file_path = os.path.join(path, filename)
        result_df.to_csv(file_path, index = False)
    return result_df

def process_csv(file_pattern, token0, token1, decimal_0, decimal_1):
    file_list = glob.glob(file_pattern)
    list_of_dfs = [pd.read_csv(f) for f in file_list]
    df_lp_total = pd.concat(list_of_dfs, ignore_index=True)
    df_lp_total = df_lp_total[df_lp_total["event"].isin(["Mint", "Burn", "Swap"])].copy()
    
    # Rename columns
    if 'amount0' in df_lp_total.columns:
        df_lp_total.rename(columns={'amount0': token0, 'amount1': token1}, inplace=True)

    # change to interger
    df_lp_total.loc[:,"timestamp"] = pd.to_numeric(df_lp_total["timestamp"])
    # change to datetime
    df_lp_total.loc[:,'datetime'] = df_lp_total['timestamp'].apply(lambda ts:datetime.fromtimestamp(ts,tz=timezone.utc))
    
    # Convert types
    cols_to_numeric = [token0, token1, "amount", "liquidity", "tick", "tickLower", "tickUpper", "blockNumber"]
    for col in cols_to_numeric:
        df_lp_total.loc[:,col] = pd.to_numeric(df_lp_total[col], errors='coerce')
    
    df_lp_total["price"] = df_lp_total["tick"].apply(lambda x: tick_2_price(x, decimal_0, decimal_1))
    df_lp_total[token0] = df_lp_total[token0].apply(lambda x: transform_amount_0(x, decimal_0))
    df_lp_total[token1] = df_lp_total[token1].apply(lambda x: transform_amount_1(x, decimal_1))
    df_lp_total['current_tick'] = df_lp_total['tick'].ffill()
    df_lp_total['current_liquidity'] = df_lp_total['liquidity'].ffill()
    df_lp_total['current_price'] = df_lp_total['price'].ffill()
    df_lp_total = df_lp_total[['datetime', 'blockNumber', 'event', token0, token1, 'amount', 'tick', "price", 'liquidity', 'tickLower', "tickUpper", 'current_tick', 'current_liquidity', 'current_price']].copy()
    return df_lp_total


def load_all_csvs(file_pattern, token0, token1):
    """
    Loads all CSV files matching the pattern, concatenates them first, 
    and THEN performs preprocessing on the combined DataFrame.
    """
    # 1. Find all files
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"Warning: No files found matching pattern '{file_pattern}'.")
        return None
    
    # 2. Read all CSVs into a list of DataFrames
    try:
        # Try using pyarrow for speed
        list_of_dfs = [pd.read_csv(f, engine='pyarrow') for f in file_list]
    except ImportError:
        print("Pyarrow not found, using default engine.")
        list_of_dfs = [pd.read_csv(f) for f in file_list]
    except Exception as e:
         print(f"Error reading CSVs: {e}. Falling back to default engine.")
         list_of_dfs = [pd.read_csv(f) for f in file_list]

    if not list_of_dfs:
        return None

    # 3. Concatenate FIRST
    df_lp_total = pd.concat(list_of_dfs, ignore_index=True)
    
    # 4. Preprocessing (on the combined DataFrame)  
    # Filter for relevant events
    if 'event' in df_lp_total.columns:
        df_lp_total = df_lp_total[df_lp_total["event"].isin(["Mint", "Burn", "Swap"])].copy()
    
    # Rename columns
    if 'amount0' in df_lp_total.columns:
        df_lp_total.rename(columns={'amount0': token0, 'amount1': token1}, inplace=True)
    
    # Convert types
    cols_to_numeric = ["timestamp", token0, token1, "amount", "liquidity", "tick", "tickLower", "tickUpper", "blockNumber"]
    for col in cols_to_numeric:
        if col in df_lp_total.columns:
            df_lp_total[col] = pd.to_numeric(df_lp_total[col], errors='coerce')
            
    # Convert datetime
    if 'timestamp' in df_lp_total.columns:
        df_lp_total['datetime'] = df_lp_total['timestamp'].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc))

    # Sort by blockNumber (Crucial for time-series logic)
    if 'blockNumber' in df_lp_total.columns:
        df_lp_total = df_lp_total.sort_values('blockNumber').reset_index(drop=True)

    # Forward fill state variables
    if 'tick' in df_lp_total.columns:
        df_lp_total['current_tick'] = df_lp_total['tick'].ffill()
    if 'liquidity' in df_lp_total.columns:
        df_lp_total['current_liquidity'] = df_lp_total['liquidity'].ffill()
        
    # Select and reorder final columns
    expected_cols = ['datetime', 'blockNumber', 'event', token0, token1, 'amount', 'tick', 'liquidity', 'tickLower', 'tickUpper', 'current_tick', 'current_liquidity']
    final_cols = [c for c in expected_cols if c in df_lp_total.columns]
    
    df_lp_total = df_lp_total[final_cols].copy()

    return df_lp_total