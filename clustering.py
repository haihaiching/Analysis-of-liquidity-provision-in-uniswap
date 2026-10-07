import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

def run_kmeans_clustering(trained_model,input_path, output_path, K=4):
    """
    Executes K-Means++ clustering on liquidity events (Mint/Burn).
    """
    # Load data
    try:
        df_all = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
        return None

    features_col = [
        'S_x_norm', 'Delta_T_x_norm', 'V_x_norm', 
        'D_L_x_norm', 'D_U_x_norm', 'L_x_norm'
    ]

    # Filter: Select rows with valid features (Mint/Burn events)
    mask = df_all[features_col].notna().all(axis=1)
    df_for_clustering = df_all[mask].copy()

    if df_for_clustering.empty:
        print("Error: No valid data for clustering.")
        return df_all

    X = df_for_clustering[features_col].values

    # Run K-Means++
    if trained_model == None:
        Model = KMeans(
            n_clusters=K,
            init='k-means++', 
            n_init=10,
            max_iter=300, 
            random_state=42
        )
    else:
        Model = trained_model

    labels = Model.fit_predict(X)

    # Assign labels back to original DataFrame
    df_all['Cluster_Label'] = np.nan
    df_all.loc[mask, 'Cluster_Label'] = labels

    # Save output
    df_all.to_csv(output_path, index=False)

    return df_all, Model


def get_v3_value(L, P, tick_lower, tick_upper, d0, d1, base_symbol):
    """
    Computes value of a V3 position given liquidity (L) and price (P).
    """
    if L <= 0 or pd.isna(L): return 0.0, 0.0, 0.0

    # Adjust for token decimals
    adj = 10 ** (int(d0) - int(d1))
    sqrt_P = np.sqrt(P / adj)
    sqrt_Pa = np.sqrt(1.0001 ** tick_lower)
    sqrt_Pb = np.sqrt(1.0001 ** tick_upper)
    
    if sqrt_Pa > sqrt_Pb: sqrt_Pa, sqrt_Pb = sqrt_Pb, sqrt_Pa
    
    x_human, y_human = 0.0, 0.0

    # Calculate token amounts based on price range
    if sqrt_P < sqrt_Pa:
        # Price below range: All in Token 0 (x)
        x_human = L * (sqrt_Pb - sqrt_Pa) / (sqrt_Pa * sqrt_Pb) / (10 ** int(d0))
    elif sqrt_P >= sqrt_Pb:
        # Price above range: All in Token 1 (y)
        y_human = L * (sqrt_Pb - sqrt_Pa) / (10 ** int(d1))
    else:
        # Price in range: Mix of both
        x_human = L * (sqrt_Pb - sqrt_P) / (sqrt_P * sqrt_Pb) / (10 ** int(d0))
        y_human = L * (sqrt_P - sqrt_Pa) / (10 ** int(d1))

    # Convert total value to base asset
    if str(base_symbol) == "0":
        total_value = y_human + (x_human * P)
    else:
        total_value = x_human + (y_human / P if P > 0 else 0)

    return total_value, x_human, y_human

def calculate_dependent_variables(df, fee_tier, decimal_0, decimal_1, token0, token1, base_symbol):
    """
    Computes returns (CONR, FRNB) and PnL metrics (HPNL, Fee, IL).
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df['hour_key'] = df['datetime'].dt.floor('h')
    
    # 1. Market Metrics: Hourly Resampling
    price_col = 'current_price'
    hourly = df.resample('1h', on='datetime').agg({price_col: ['first', 'last']})
    hourly.columns = ['open', 'close']
    hourly = hourly.ffill().bfill()

    # Log Returns & Future Returns
    hourly['conr'] = np.log(hourly['close'] / hourly['open'])
    hourly['frnb'] = hourly['conr'].shift(-1)
    
    df['CONR'] = df['hour_key'].map(hourly['conr'])
    df['FRNB'] = df['hour_key'].map(hourly['frnb'])

    # 2. Fee Estimation
    fee_mult = float(fee_tier) / (1 - float(fee_tier))
    vol_col = token1 if str(base_symbol) == '0' else token0
    
    swaps = df[df['event'] == 'Swap'].copy()
    swaps['fee_val'] = swaps[vol_col].abs() * fee_mult
    if vol_col == token0: 
        swaps['fee_val'] *= swaps['current_price']
    
    # Group swaps for fast lookup
    swaps_grouped = {k: v for k, v in swaps[swaps['current_liquidity'] > 0].groupby('hour_key')}

    # 3. HPNL Calculation (Fee + IL)
    results = {}
    lp_events = df[df['event'].isin(['Mint', 'Burn'])]

    for idx, row in lp_events.iterrows():
        L, t_ev, p_en = row['amount'], row['datetime'], row['current_price']
        t_l, t_u, h_key = row['tickLower'], row['tickUpper'], row['hour_key']
        next_h = h_key + pd.Timedelta(hours=1)
        
        if L <= 0:
            results[idx] = {'HPNL': 0.0, 'HPNL_Fee': 0.0, 'HPNL_IL': 0.0}
            continue
        
        # Settlement price (next hour close)
        p_ex = hourly.loc[next_h, 'close'] if next_h in hourly.index else hourly['close'].asof(next_h)
        if pd.isna(p_ex): continue

        # A. Accrue Fees
        f_sum = 0.0
        for h in [h_key, next_h]:
            s_data = swaps_grouped.get(h, pd.DataFrame())
            if s_data.empty: continue
            
            # Filter relevant swaps (time & tick range)
            time_mask = (s_data['datetime'] > t_ev) if h == h_key else True
            in_range = s_data['tick'].between(t_l, t_u)
            valid = s_data[time_mask & in_range]
            
            if not valid.empty:
                f_sum += ((valid['fee_val'] / valid['current_liquidity']) * L).sum()

        # B. Calculate Impermanent Loss
        v_en, a0_en, a1_en = get_v3_value(L, p_en, t_l, t_u, decimal_0, decimal_1, base_symbol)
        v_ex, _, _ = get_v3_value(L, p_ex, t_l, t_u, decimal_0, decimal_1, base_symbol)
        v_hodl = a1_en + a0_en * p_ex 
        
        il_val = v_ex - v_hodl

        results[idx] = {
            'HPNL': (f_sum + il_val) / L,
            'HPNL_Fee': f_sum / L,
            'HPNL_IL': il_val / L
        }

    # Merge results
    res_df = pd.DataFrame.from_dict(results, orient='index')
    df = pd.concat([df, res_df], axis=1)

    return df.drop(columns=['hour_key'])

def calculate_cluster_net_liquidity_flow(df, time_col, cluster_col, amount_col, type_col, freq):
    """
    Computes Net Liquidity Flow (NLF) per cluster.
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.dropna(subset=[cluster_col])
    
    # Sign Liquidity: Mint (+), Burn (-)
    data['signed_liquidity'] = 0.0
    event_series = data[type_col].astype(str).str.lower()

    mint_mask = event_series.str.contains('mint', regex=True)
    data.loc[mint_mask, 'signed_liquidity'] = data.loc[mint_mask, amount_col].abs()
    
    burn_mask = event_series.str.contains('burn', regex=True)
    data.loc[burn_mask, 'signed_liquidity'] = -data.loc[burn_mask, amount_col].abs()
    
    # Aggregate by Time & Cluster
    grouped = data.groupby(
        [pd.Grouper(key=time_col, freq=freq), cluster_col]
    )['signed_liquidity'].sum()
    
    if grouped.empty:
        return pd.DataFrame(columns=[time_col, cluster_col, 'NLF'])

    # Reindex to fill missing time intervals with 0
    min_time = grouped.index.get_level_values(time_col).min()
    max_time = grouped.index.get_level_values(time_col).max()
    all_clusters = data[cluster_col].dropna().unique()
    
    full_idx = pd.MultiIndex.from_product(
        [pd.date_range(min_time, max_time, freq=freq, name=time_col), all_clusters], 
        names=[time_col, cluster_col]
    )
    
    flow_df = grouped.reindex(full_idx, fill_value=0).reset_index()
    flow_df.rename(columns={'signed_liquidity': 'NLF'}, inplace=True)
    
    return flow_df

def analyze_and_plot_correlations(raw_df: pd.DataFrame, time_col, cluster_col, amount_col, type_col, freq):
    """
    Runs the full analysis pipeline:
    1. Calculates Net Liquidity Flow (NLF) from raw data.
    2. Analyzes correlations between NLF and Metrics (CONR, FRNB, HPNL).
    3. Plots the correlation heatmap.
    """
    # Calculating Net Liquidity Flow
    nlf_df = calculate_cluster_net_liquidity_flow(
        raw_df, time_col, cluster_col, amount_col, type_col, freq
    )

    # Aggregation rules: HPNL (Sum), Others (Mean)
    agg_rules = {'CONR': 'mean', 'FRNB': 'mean', 'HPNL': 'sum'}
    
    raw_metrics = raw_df.copy()
    raw_metrics[time_col] = pd.to_datetime(raw_metrics[time_col])
    raw_metrics = raw_metrics.dropna(subset=[cluster_col])
    
    existing_cols = [col for col in agg_rules.keys() if col in raw_metrics.columns]
    final_rules = {k: agg_rules[k] for k in existing_cols}
    
    if not existing_cols: return None

    # Aggregate metrics to hourly
    metrics_hourly = raw_metrics.groupby(
        [pd.Grouper(key=time_col, freq='1H'), cluster_col]
    ).agg(final_rules).reset_index()
    
    # Merge NLF with Metrics
    nlf_df[time_col] = pd.to_datetime(nlf_df[time_col])
    df_merged = pd.merge(nlf_df, metrics_hourly, on=[time_col, cluster_col], how='inner')
    
    if df_merged.empty: return None

    # Compute Correlations
    corr_results = []
    unique_clusters = sorted(df_merged[cluster_col].unique())
    
    for cluster in unique_clusters:
        sub_df = df_merged[df_merged[cluster_col] == cluster]
        if len(sub_df) < 2: continue
            
        corrs = sub_df[['NLF'] + existing_cols].corr()['NLF'].drop('NLF')
        row = corrs.to_dict()
        row[cluster_col] = cluster
        corr_results.append(row)
        
    corr_df = pd.DataFrame(corr_results).set_index(cluster_col)
    
    target_order = [c for c in ['CONR', 'FRNB', 'HPNL'] if c in corr_df.columns]
    corr_df = corr_df[target_order]
    
    # Plot Heatmap
    plt.figure(figsize=(8, 5))
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    
    y_labels = [rf'$NLF(\phi_{int(c)})$' for c in corr_df.index]
    
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="YlOrRd", 
        vmin=-0.1, vmax=0.5, linewidths=.5, cbar_kws={'label': 'Correlation'}
    )
    
    plt.title("Correlation Analysis (Hourly)", fontweight='bold', pad=12)
    plt.yticks(ticks=[i + 0.5 for i in range(len(corr_df))], labels=y_labels, rotation=0)
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    
    return corr_df





def analyze_and_plot_robust_correlations(
    raw_df: pd.DataFrame, 
    time_col='datetime', 
    cluster_col='Cluster_Label',
    amount_col='amount',
    type_col='type',
    freq='1H'
):
    """
    Analyzes robust correlations (Spearman) between active NLF (non-zero) and Metrics.
    This version filters out periods where NLF is 0 to focus on active decision-making.
    """
    nlf_df = calculate_cluster_net_liquidity_flow(
        raw_df, time_col, cluster_col, amount_col, type_col, freq
    )
    # 1. Define Aggregation Rules
    # HPNL uses 'sum', others use 'first' (since they are constant per hour)
    agg_rules = {'CONR': 'first', 'FRNB': 'first', 'HPNL': 'sum'}
    
    raw_metrics = raw_df.copy()
    raw_metrics[time_col] = pd.to_datetime(raw_metrics[time_col])
    raw_metrics = raw_metrics.dropna(subset=[cluster_col])
    
    existing_cols = [col for col in agg_rules.keys() if col in raw_metrics.columns]
    final_rules = {k: agg_rules[k] for k in existing_cols}
    
    # Aggregate metrics to hourly
    metrics_hourly = raw_metrics.groupby(
        [pd.Grouper(key=time_col, freq='1H'), cluster_col]
    ).agg(final_rules).reset_index()
    
    # 2. Merge NLF with Metrics
    nlf_df = nlf_df.copy()
    nlf_df[time_col] = pd.to_datetime(nlf_df[time_col])
    
    df_merged = pd.merge(nlf_df, metrics_hourly, on=[time_col, cluster_col], how='inner')
    
    if df_merged.empty:
        print("Error: Merged data is empty.")
        return None

    # 3. Compute Robust Correlations
    corr_results = []
    unique_clusters = sorted(df_merged[cluster_col].unique())
    
    print("\n[Data Diagnostics]")
    for cluster in unique_clusters:
        sub_df = df_merged[df_merged[cluster_col] == cluster]
        
        # --- Critical Step: Filter out zero NLF ---
        # We only want to see if the direction of liquidity flow correlates with metrics
        # when the cluster is actually doing something.
        active_df = sub_df[sub_df['NLF'] != 0]
        
        n_total = len(sub_df)
        n_active = len(active_df)
        print(f"Cluster {cluster}: Total Hours {n_total}, Active Hours {n_active} ({n_active/n_total:.1%})")
        
        if len(active_df) < 5: # Skip if too few active data points
            print(f"  -> Too few active hours, skipping.")
            continue
            
        # --- Critical Step: Use Spearman Correlation ---
        # Spearman is rank-based and handles outliers/extreme values better than Pearson.
        corrs = active_df[['NLF'] + existing_cols].corr(method='spearman')['NLF'].drop('NLF')
        
        row = corrs.to_dict()
        row[cluster_col] = cluster
        corr_results.append(row)
        
    if not corr_results:
        print("No sufficient active data to compute correlations.")
        return None

    corr_df = pd.DataFrame(corr_results).set_index(cluster_col)
    target_order = [c for c in ['CONR', 'FRNB', 'HPNL'] if c in corr_df.columns]
    corr_df = corr_df[target_order]
    
    # 4. Plot Heatmap
    plt.figure(figsize=(8, 5))
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    
    y_labels = [rf'$NLF(\phi_{int(c)})$' for c in corr_df.index]
    
    sns.heatmap(
        corr_df, 
        annot=True, 
        fmt=".2f", 
        cmap="RdBu_r",  # Red-Blue colormap is better for Spearman (shows +/- distinctively)
        center=0,       # Center the color map at 0 (white)
        vmin=-0.3, vmax=0.3, # Narrower range to highlight weaker correlations
        linewidths=.5,
        cbar_kws={'label': 'Spearman Correlation (Active Only)'}
    )
    
    plt.title("Active Periods Correlation (Spearman)", fontweight='bold', pad=12)
    plt.yticks(ticks=[i + 0.5 for i in range(len(corr_df))], labels=y_labels, rotation=0)
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    
    return corr_df


