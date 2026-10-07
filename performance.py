import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
from scipy import stats

def calculate_sharpe_and_plot(raw_df, time_col, cluster_col, hpnl_col, target_vol, risk_free_rate=0.0):
    """
    Calculates the annualized Sharpe Ratio and plots cumulative PnL.
    Rescales returns to a target volatility and compares clusters against a benchmark.
    """
    
    # 1. Data preprocessing: Time conversion and date extraction
    data = raw_df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data['date'] = data[time_col].dt.date
    
    # 2. Aggregate Daily HPNL by Cluster
    # Sum daily HPNL and pivot to Date-Cluster matrix
    daily_cluster = data.groupby(['date', cluster_col])[hpnl_col].sum().reset_index()
    daily_pivot = daily_cluster.pivot(index='date', columns=cluster_col, values=hpnl_col).fillna(0)
    
    # 3. Calculate "No Cluster" Benchmark
    # Aggregate total HPNL per day regardless of cluster
    daily_benchmark = data.groupby('date')[hpnl_col].sum()
    daily_pivot['No Cluster'] = daily_benchmark
    
    # 4. Volatility Rescaling and Sharpe Ratio Computation
    daily_pivot.index = pd.to_datetime(daily_pivot.index)
    results = {}
    rescaled_series = pd.DataFrame(index=daily_pivot.index)
    
    # Annualization factor for crypto markets (365 days)
    annual_factor = np.sqrt(365)
    
    for col in daily_pivot.columns:
        series = daily_pivot[col]
        daily_std = series.std()
        
        # Avoid division by zero for inactive clusters
        if daily_std == 0:
            print(f"⚠️ Warning: Cluster {col} has zero volatility; skipping SR calculation.")
            continue
            
        # Calculate scaling factor to reach target annualized volatility
        # Factor = target_vol / (daily_std * sqrt(365))
        scaling_factor = target_vol / (daily_std * annual_factor)
        
        # Apply rescaling
        rescaled = series * scaling_factor
        rescaled_series[col] = rescaled
        
        # Calculate Annualized Sharpe Ratio
        # SR = (Mean / Std) * sqrt(365)
        mean_rescaled = rescaled.mean()
        std_rescaled = rescaled.std()
        sr = (mean_rescaled / std_rescaled) * annual_factor
        results[col] = sr

    # 5. Visualization: Cumulative Returns
    # Convert rescaled PnL to cumulative percentage
    cumulative_pnl = rescaled_series.cumsum() * 100 
    
    # Set plotting aesthetics
    sns.set_context("paper", font_scale=1.4)
    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.figure(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(cumulative_pnl.columns))
    best_cluster = max(results, key=results.get)
    
    for i, col in enumerate(cumulative_pnl.columns):
        sr_val = results[col]
        label_text = f"{col}: SR = {sr_val:.2f}"
        
        # Apply specific styles for benchmark and best-performing cluster
        if col == 'No Cluster':
            line_style = '--'
            color = 'grey'
            alpha = 0.8
            width = 2
        else:
            line_style = '-' 
            color = colors[i]
            alpha = 1.0
            width = 2.5 if col == best_cluster else 1.5
            
        plt.plot(
            cumulative_pnl.index, 
            cumulative_pnl[col], 
            label=label_text,
            linestyle=line_style,
            color=color,
            linewidth=width,
            alpha=alpha
        )

    # Chart formatting
    plt.title(f"Cumulative HPNL (Rescaled to {target_vol:.0%} Volatility)", fontweight='bold', pad=15)
    plt.xlabel("Date", fontweight='bold')
    plt.ylabel("Cumulative Return (%)", fontweight='bold')
    plt.legend(frameon=True, loc='upper left')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
    return pd.Series(results), rescaled_series


def plot_pca_clusters(df, features_col=None, cluster_col='Cluster_Label'):
    """
    Performs PCA to reduce feature dimensions to 2D and plots the clusters.
    """
    
    # 1. Define Features (Default to the ones used in clustering)
    if features_col is None:
        features_col = [
            'S_x_norm', 'Delta_T_x_norm', 'V_x_norm', 
            'D_L_x_norm', 'D_U_x_norm', 'L_x_norm'
        ]
    
    # 2. Prepare Data
    # Ensure we only use rows with valid labels and features
    mask = df[features_col + [cluster_col]].notna().all(axis=1)
    df_pca = df[mask].copy()
    
    if df_pca.empty:
        return None

    X = df_pca[features_col].values
    y = df_pca[cluster_col].values

    # 3. Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Add PCA results back to DataFrame for plotting
    df_pca['PC1'] = X_pca[:, 0]
    df_pca['PC2'] = X_pca[:, 1]
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_


    # 4. Plotting
    plt.figure(figsize=(10, 7))
    sns.set_context("paper", font_scale=1.2)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Scatter plot
    sns.scatterplot(
        data=df_pca, 
        x='PC1', 
        y='PC2', 
        hue=cluster_col, 
        palette='viridis', 
        alpha=0.7,
        s=25, # Marker size
        edgecolor='w'
    )
    
    plt.title(f'PCA Visualization of LP Clusters (Total Var: {explained_var.sum():.1%})', fontweight='bold', pad=15)
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1%})')
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1%})')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return df_pca

def analyze_optimal_k(input_path, k_min=2, k_max=8, model_type='kmeans'):
    """
    Evaluates clustering performance for a range of K values.
    Calculates Inertia/BIC and Silhouette scores to identify the optimal number of clusters.
    """
    # 1. Load dataset and define feature columns
    df = pd.read_csv(input_path)
    features_col = ['S_x_norm', 'Delta_T_x_norm', 'V_x_norm', 'D_L_x_norm', 'D_U_x_norm', 'L_x_norm']
    
    # 2. Convert features to numeric types
    for col in features_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Extract feature matrix
    X = df[features_col].values
    
    # 4. Handle missing or infinite values for numerical stability
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Initialize evaluation metrics
    metrics = []      # Stores Inertia (KMeans) or BIC (GMM)
    silhouettes = []  # Stores Silhouette scores
    K_range = range(k_min, k_max + 1)
    
    # Downsample large datasets to improve computation speed
    if len(X) > 10000:
        rng = np.random.default_rng(42)
        X_sample = rng.choice(X, 10000, axis=0, replace=False)
    else:
        X_sample = X

    # Iterate through K values to fit models
    for k in K_range:
        try:
            if model_type == 'gmm':
                # Fit Gaussian Mixture Model and compute BIC
                model = GaussianMixture(n_components=k, random_state=42, n_init=3)
                labels = model.fit_predict(X_sample)
                metric = model.bic(X_sample)
            else: 
                # Fit KMeans and compute Inertia
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_sample)
                metric = model.inertia_
            
            # Compute Silhouette score for clustering quality
            score = silhouette_score(X_sample, labels)
            
            metrics.append(metric)
            silhouettes.append(score)
            
        except Exception:
            metrics.append(np.nan)
            silhouettes.append(np.nan)

    # 6. Plot performance metrics
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    metric_name = 'BIC (Lower is better)' if model_type == 'gmm' else 'Inertia (Lower is better)'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel(metric_name, color=color, fontweight='bold')
    ax1.plot(K_range, metrics, 'o-', color=color, lw=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette (Higher is better)', color=color, fontweight='bold')
    ax2.plot(K_range, silhouettes, 's--', color=color, lw=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Optimal K Analysis ({model_type.upper()})')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_clusters(input_path, random_state=42):
    """
    Generates 2D visualizations of clustering results using t-SNE and UMAP.
    Adjusts hyper-parameters dynamically based on the dataset size.
    """
    # 1. Load data with error handling
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        return

    # 2. Validate required columns
    features_col = ['S_x_norm', 'Delta_T_x_norm', 'V_x_norm', 'D_L_x_norm', 'D_U_x_norm', 'L_x_norm']
    if not all(col in df.columns for col in features_col):
        return

    # Identify existing cluster label column
    if 'Cluster' in df.columns:
        label_col = 'Cluster'
    elif 'Cluster_Label' in df.columns:
        label_col = 'Cluster_Label'
    else:
        return

    # 3. Apply optional filtering for liquidity events
    if 'event' in df.columns:
        df['event_clean'] = df['event'].astype(str).str.lower().str.strip()
        df_filtered = df[df['event_clean'].isin(['mint', 'burn'])].copy()
        
        # Fallback if filtered result is empty
        df_final = df_filtered if len(df_filtered) > 0 else df
    else:
        df_final = df

    # 4. Prepare data matrix and labels
    X = df_final[features_col].values
    labels = df_final[label_col].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_samples = len(X)
    if n_samples < 2:
        return

    # 5. Initialize plotting layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- t-SNE Projection ---
    # Set perplexity dynamically based on sample size
    safe_perplexity = min(30, max(5, n_samples - 1))
    
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', 
                perplexity=safe_perplexity, random_state=random_state)
    embedding_tsne = tsne.fit_transform(X)
    
    # Scale point size and transparency for visual clarity
    pt_size = 50 if n_samples < 100 else (5 if n_samples < 10000 else 1)
    pt_alpha = 0.8 if n_samples < 100 else (0.6 if n_samples < 10000 else 0.3)

    scatter1 = axes[0].scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], 
                               c=labels, cmap='viridis', s=pt_size, alpha=pt_alpha)
    axes[0].set_title(f't-SNE (Perplexity={safe_perplexity})', fontsize=14)
    axes[0].set_xlabel('Dim 1')
    axes[0].set_ylabel('Dim 2')
    fig.colorbar(scatter1, ax=axes[0], label='Cluster')

    # --- UMAP Projection ---
    if umap is not None:
        # Set neighbors dynamically based on sample size
        safe_neighbors = min(30, max(2, n_samples - 1))
        
        reducer = umap.UMAP(n_neighbors=safe_neighbors, min_dist=0.1, random_state=random_state)
        embedding_umap = reducer.fit_transform(X)
        
        scatter2 = axes[1].scatter(embedding_umap[:, 0], embedding_umap[:, 1], 
                                   c=labels, cmap='viridis', s=pt_size, alpha=pt_alpha)
        axes[1].set_title(f'UMAP (Neighbors={safe_neighbors})', fontsize=14)
        axes[1].set_xlabel('Dim 1')
        axes[1].set_ylabel('Dim 2')
        fig.colorbar(scatter2, ax=axes[1], label='Cluster')
    else:
        axes[1].text(0.5, 0.5, "UMAP not installed", ha='center', va='center', fontsize=14)
        axes[1].set_title("UMAP Projection")

    plt.tight_layout()
    plt.show()


def analyze_cluster_performance(raw_df, time_col, cluster_col, hpnl_col, target_vol):
    """
    Computes performance metrics based on daily aggregated HPNL following specific methodology.
    
    Steps:
    1. Calculate Total HPNL for each cluster per day.
    2. Rescale series to the target volatility (default σ_tgt = 0.15).
    3. Compute Annualized Sharpe Ratio, Sortino Ratio, and Calmar Ratio.
    
    Formulas:
    - HPNL_total(φ_i, T) = Σ HPNL(φ_i, Δ_j, T) for all events j on day T.
    - Rescaled_HPNL = (σ_tgt / (STD_daily * √365)) * HPNL_daily
    - Annualized_SR = (E[Rescaled] / STD(Rescaled)) * √365
    
    Args:
        raw_df: DataFrame containing transaction logs.
        time_col: Name of the timestamp column.
        cluster_col: Name of the cluster label column.
        hpnl_col: Name of the HPNL column.
        target_vol: Target annualized volatility (default 0.15).
    
    Returns:
        metrics_df: DataFrame containing calculated metrics for all clusters and benchmark.
    """
    
    # Filter for liquidity provision events and prepare time features
    data = raw_df[raw_df['event'].isin(['Mint', 'Burn'])].copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data['date'] = data[time_col].dt.date
    
    # Identify unique clusters
    cluster_labels = sorted(data[cluster_col].unique())
    all_labels = list(cluster_labels) + ['No Cluster']
    
    # Sum HPNL per cluster per day
    daily_cluster_hpnl = data.groupby(['date', cluster_col])[hpnl_col].sum().reset_index()
    daily_pivot = daily_cluster_hpnl.pivot(index='date', columns=cluster_col, values=hpnl_col).fillna(0)
    
    # Add 'No Cluster' benchmark representing the aggregate of all transactions
    daily_pivot['No Cluster'] = data.groupby('date')[hpnl_col].sum()

    
    results = []
    rescaled_series_dict = {}
    
    for col in daily_pivot.columns:
        
        # Select daily series for the current cluster/benchmark
        daily_hpnl = daily_pivot[col]


        # Scaling factor is calculated to normalize annualized volatility to target_vol
        daily_std = daily_hpnl.std()
        
        if daily_std == 0:
            print(f"Warning: Daily Std = 0 for cluster {col}, skipping rescaling")
            scaling_factor = 0
            rescaled_hpnl = daily_hpnl * 0
        else:
            # Formula: scaling_factor = target_vol / (daily_std * √365)
            scaling_factor = target_vol / (daily_std * np.sqrt(365))
            
            # Apply scaling factor
            rescaled_hpnl = daily_hpnl * scaling_factor
        
        rescaled_series_dict[col] = rescaled_hpnl
        
        # Annualized Sharpe Ratio
        # SR = (Mean Daily / Std Daily) * √365
        mean_rescaled = rescaled_hpnl.mean()
        std_rescaled = rescaled_hpnl.std()
        
        if std_rescaled != 0:
            sharpe_ratio = (mean_rescaled / std_rescaled) * np.sqrt(365)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (Downside deviation only)
        downside_returns = rescaled_hpnl[rescaled_hpnl < 0]
        
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns**2))
            sortino_ratio = (mean_rescaled / downside_std) * np.sqrt(365)
        else:
            sortino_ratio = np.inf
        
        # Maximum Drawdown based on cumulative rescaled PnL
        cum_returns = rescaled_hpnl.cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns - running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        # Annualized Return divided by absolute Maximum Drawdown
        annualized_return = mean_rescaled * 365
        
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = np.inf
        
        # Win Rate (Positive daily PnL ratio)
        win_rate = (rescaled_hpnl > 0).sum() / len(rescaled_hpnl)
        
        # Summary returns
        total_return = rescaled_hpnl.sum()
        annualized_return_alt = mean_rescaled * 365
        
        # Store results for the current cluster
        results.append({
            'Cluster': col,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Win Rate (%)': win_rate * 100,
        })
    
    # Create final metrics table
    metrics_df = pd.DataFrame(results).set_index('Cluster')
    
    # Output formatted summary table
    print(f"\n{'Cluster':<15} {'Sharpe':<10} {'Sortino':<10} {'Calmar':<10} {'Max DD':<10} {'Win Rate':<10}")
    for idx, row in metrics_df.iterrows():
        print(f"{idx:<15} "
              f"{row['Sharpe Ratio']:>8.2f}  "
              f"{row['Sortino Ratio']:>8.2f}  "
              f"{row['Calmar Ratio']:>8.2f}  "
              f"{row['Max Drawdown (%)']:>7.1f}%  "
              f"{row['Win Rate (%)']:>7.1f}%")
    
    return metrics_df