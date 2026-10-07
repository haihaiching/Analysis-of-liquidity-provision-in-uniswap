import pandas as pd
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import time
from bisect import bisect_left, bisect_right

def tick_2_price(tick, decimal_0, decimal_1):
    return (1.0001 ** tick) * (10 ** (int(decimal_0) - int(decimal_1))) 

def transform_amount_0(amount, decimal_0):
    return amount / 10 ** int(decimal_0)

def transform_amount_1(amount, decimal_1):
    return amount / 10 ** int(decimal_1)

def calculate_features(df_lp, token0, token1, decimal_0, decimal_1, tickspacing, base_symbol):
    """
    Calculates S_x, Delta_T_x, V_x, D_L_x, D_U_x.
    """
    tickspacing_num = int(tickspacing)

    # S_x
    print("Calculating S(x)")
    p_tx = df_lp['current_tick']
    k1_pi = df_lp['tickLower']   
    k2_pi = df_lp['tickUpper']
    is_lp_event = df_lp['event'].isin(['Mint', 'Burn'])
    in_the_money = (k1_pi <= p_tx) & (p_tx < k2_pi) & is_lp_event
    df_lp['S_x'] = np.where(
        is_lp_event & df_lp['current_liquidity'] > 0, 
        (df_lp['amount']*in_the_money) / df_lp['current_liquidity'], 
        np.nan
    )
    
    # Delta_T_x & V_x
    print("Calculating Delta_T(x) & V(x)")
    df_lp['Delta_T_x'] = np.nan
    df_lp['V_x'] = np.nan
    last_tick_spacing_group = None
    last_tick_spacing_move_time = pd.NaT
    cumulative_volume_since_move = 0.0

    for row in df_lp.itertuples():
        if row.event == 'Swap':
            if pd.isna(row.tick):
                continue
            current_tick_spacing_group = int(row.tick // tickspacing_num)
            if last_tick_spacing_group is None or \
                current_tick_spacing_group != last_tick_spacing_group:
                last_tick_spacing_group = current_tick_spacing_group
                last_tick_spacing_move_time = row.datetime
                cumulative_volume_since_move = 0.0 
            
            if base_symbol == "0":
                cumulative_volume_since_move += abs(getattr(row, token1))
            elif base_symbol == "1":
                cumulative_volume_since_move += abs(getattr(row, token0))
                
        elif row.event == 'Mint' or row.event == 'Burn':
            if pd.notna(last_tick_spacing_move_time):
                time_diff = row.datetime - last_tick_spacing_move_time
                df_lp.at[row.Index, 'Delta_T_x'] = time_diff.total_seconds()
                df_lp.at[row.Index, 'V_x'] = cumulative_volume_since_move
                
    # D_L_x, D_U_x
    print("Calculating D_L(x) & D_U(x)")
    p_tx = df_lp['current_tick']
    k1_pi = df_lp['tickLower']   
    k2_pi = df_lp['tickUpper']
    is_lp_event = df_lp['event'].isin(['Mint', 'Burn'])
    in_the_money = (k1_pi <= p_tx) & (p_tx < k2_pi) & is_lp_event
    D_L = k1_pi - p_tx
    D_U = k2_pi - p_tx
    df_lp['D_L_x'] = np.where(is_lp_event, np.where(in_the_money, 0, D_L), np.nan)
    df_lp['D_U_x'] = np.where(is_lp_event, np.where(in_the_money, 0, D_U), np.nan)
    df_lp.to_csv("data_feature\data_features.csv", index = False)
    return df_lp

def LP_position(event, amount, tickLower, tickUpper, tick_min, tick_max):
    ticks = np.arange(tick_min, tick_max + 1, 10)
    indicator = ((ticks >= tickLower) & (ticks < tickUpper)).astype(float)
    
    if event == 'Mint':
        return amount * indicator 
    
    if event == 'Burn':
        return -amount * indicator 
    
    else:
        return 0 * indicator 

def tick_2_raw_sqrt_price(tick):
    return math.sqrt(1.0001 ** tick)


def get_Lx(df, tick_min, tick_max, tickspacing, decimal_0, decimal_1, base_symbol):
    # State tracking for liquidity distribution and active price points
    liquidity_by_tick = defaultdict(float)
    sorted_active_ticks = []
    lx_results = {}

    # Cast parameters to localized types for faster access within the loop
    decimal_0_int = int(decimal_0)
    decimal_1_int = int(decimal_1)
    tickspacing_int = int(tickspacing)
    base_symbol_flag = (base_symbol == "0")
    
    # Memoization for square root calculations to reduce repetitive floating point operations
    sqrt_cache = {}
    def get_sqrt(tick):
        if tick not in sqrt_cache:
            sqrt_cache[tick] = math.sqrt(1.0001 ** tick)
        return sqrt_cache[tick]

    # Cache structure to store and validate the previous calculation state
    cache = {
        'k_p': None,
        'k1_x': None, 
        'k2_x': None,
        'result': 0.0,
        'valid': False
    }
    
    start_time = time.time()
    
    for row in df.itertuples():
        idx = row.Index
        
        # Determine if the price state is calculable
        if pd.isna(row.current_tick):
            lx_results[idx] = 0.0
            cache['valid'] = False
        else:
            k_p = int(row.current_tick)
            k1_x = int(row.tickLower)
            k2_x = int(row.tickUpper)
            
            # Retrieve from cache if price parameters and boundaries remain unchanged
            if (cache['valid'] and 
                k_p == cache['k_p'] and
                k1_x == cache['k1_x'] and
                k2_x == cache['k2_x']):
                lx_results[idx] = cache['result']
            else:
                total_value_usd = 0.0
                
                # Liquidity is considered inactive if the current price falls within the position range
                if k1_x <= k_p < k2_x:
                    total_value_usd = 0.0
                else:
                    # Isolate active liquidity ticks using binary search for $O(\log N)$ range selection
                    if k1_x > k_p:
                        # Price is below the range: find ticks in $[k_p, k1_x)$
                        start_idx = bisect_left(sorted_active_ticks, k_p)
                        end_idx = bisect_left(sorted_active_ticks, k1_x)
                        relevant_ticks = sorted_active_ticks[start_idx:end_idx]
                    else: 
                        # Price is above the range: find ticks in $(k2_x, k_p]$
                        start_idx = bisect_right(sorted_active_ticks, k2_x)
                        end_idx = bisect_right(sorted_active_ticks, k_p)
                        relevant_ticks = sorted_active_ticks[start_idx:end_idx]
                    
                    if len(relevant_ticks) > 0:
                        sqrt_P_current = get_sqrt(k_p)
                        # Scaling factor based on token decimals
                        P_current = (sqrt_P_current ** 2) * (10 ** (decimal_0_int - decimal_1_int))
                        
                        for tick in relevant_ticks:
                            l_val = liquidity_by_tick[tick]
                            if l_val <= 0:
                                continue
                            
                            sqrt_P_a = get_sqrt(tick)
                            sqrt_P_b = get_sqrt(tick + tickspacing_int)
                            
                            amount_0 = 0.0
                            amount_1 = 0.0
                            
                            # Apply Uniswap v3 liquidity-to-token conversion formulas
                            if k_p >= tick + tickspacing_int:
                                amount_1 = l_val * (sqrt_P_b - sqrt_P_a) / (10 ** decimal_1_int)
                            elif k_p < tick:
                                amount_0 = l_val * (1/sqrt_P_a - 1/sqrt_P_b) / (10 ** decimal_0_int)
                            
                            # Normalize value to the selected base currency
                            if base_symbol_flag:
                                total_value_usd += amount_1 + (amount_0 * P_current)
                            else:
                                total_value_usd += amount_0 + (amount_1 / P_current if P_current != 0 else 0)
                
                lx_results[idx] = total_value_usd
                
                # Update cache with the latest computed parameters
                cache.update({
                    'k_p': k_p,
                    'k1_x': k1_x,
                    'k2_x': k2_x,
                    'result': total_value_usd,
                    'valid': True
                })

        # Process state-changing events (Mint/Burn)
        tick_lower = int(row.tickLower)
        tick_upper = int(row.tickUpper)
        amount = float(row.amount)
        
        if row.event == 'Mint':
            for tick in range(tick_lower, tick_upper, tickspacing_int):
                was_zero = (liquidity_by_tick[tick] == 0)
                liquidity_by_tick[tick] += amount
                # Insert new active ticks into the sorted list to maintain $O(\log N)$ searchability
                if was_zero and liquidity_by_tick[tick] > 0:
                    pos = bisect_left(sorted_active_ticks, tick)
                    sorted_active_ticks.insert(pos, tick)
                    
        elif row.event == 'Burn':
            for tick in range(tick_lower, tick_upper, tickspacing_int):
                liquidity_by_tick[tick] -= amount
                if liquidity_by_tick[tick] < 1e-10:
                    liquidity_by_tick[tick] = 0.0
                    try:
                        pos = sorted_active_ticks.index(tick)
                        sorted_active_ticks.pop(pos)
                    except ValueError:
                        pass
        
        # Invalidate cache if the current liquidity change overlaps with the range used for the cached result
        if cache['valid']:
            if cache['k1_x'] > cache['k_p']:
                if not (tick_upper <= cache['k_p'] or tick_lower >= cache['k1_x']):
                    cache['valid'] = False
            elif cache['k2_x'] < cache['k_p']:
                if not (tick_upper <= cache['k2_x'] or tick_lower > cache['k_p']):
                    cache['valid'] = False
                    
    return pd.Series(lx_results, name='L_x')

def calculate_Lx(df_lp_total, token0, token1, decimal_0, decimal_1, tickspacing, base_symbol):
    """
    Wraps step 4: Filters LP events and calculates L(x).
    """
    print("Calculating L(x)")
    df_lp_events = df_lp_total[df_lp_total['event'].isin(['Mint', 'Burn'])].copy()

    df_lp_events = df_lp_events.sort_values('blockNumber')
    min_tick = int(df_lp_events['tickLower'].min())
    max_tick = int(df_lp_events['tickUpper'].max())
    lx_series = get_Lx(df_lp_events, min_tick, max_tick, tickspacing, decimal_0, decimal_1, base_symbol)

    df_lp_total.loc[lx_series.index, 'L_x'] = lx_series
    df_lp_total.to_csv("data_feature/data_total.csv", index = False)
    return df_lp_total


def normalize_and_save(df_lp_features, start_date_str, end_date_str, output_file):
    """
    Wraps step 5: Saves raw data, normalizes features, filters by date, and saves final output.
    """
    features_to_normalize = ['S_x', 'Delta_T_x', 'V_x', 'D_L_x', 'D_U_x', 'L_x']
    window_size = 100 
    df_result = df_lp_features.copy()

    mask = df_result['event'].isin(['Mint', 'Burn'])
    df_subset = df_result[mask].copy()

    rolling_mean = df_subset[features_to_normalize].rolling(window=window_size).mean()
    rolling_std = df_subset[features_to_normalize].rolling(window=window_size).std()

    # N(X, t) = (X(t) - MA) / STD
    df_normalized = (df_subset[features_to_normalize] - rolling_mean) / (rolling_std)
    df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)
    df_normalized.columns = [f'{col}_norm' for col in df_normalized.columns]

    for col in df_normalized.columns:
        df_result[col] = np.nan
        df_result.loc[mask, col] = df_normalized[col]
    
    df_result['datetime'] = pd.to_datetime(df_result['datetime'])
    start_date, end_date = pd.to_datetime([start_date_str, end_date_str])
    
    df_final = df_result[
        (df_result['datetime'] >= start_date) & 
        (df_result['datetime'] < end_date)
    ].copy()

    df_final.to_csv(output_file, index = False)
    return df_final 
    