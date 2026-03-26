import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def plot_feature_distributions(X: pl.DataFrame, figsize=(20, 18)):
    """
    Plot distribution of all 15 features, one histogram per feature.
    Excludes the agent column.
    """
    feature_cols = [c for c in X.columns if c != "agent"]
    n_features   = len(feature_cols)
    n_cols       = 3
    n_rows       = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes      = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax     = axes[i]
        values = X[col].drop_nulls().to_numpy()
        values = values[np.isfinite(values)]  # drop nan/inf

        ax.hist(values, bins=50, edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("count")

        # annotate basic stats
        ax.axvline(np.median(values), color="red",    linestyle="--", linewidth=1, label=f"median={np.median(values):.2g}")
        ax.axvline(np.mean(values),   color="orange", linestyle="--", linewidth=1, label=f"mean={np.mean(values):.2g}")
        ax.legend(fontsize=7)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions (raw)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("output/feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/feature_distributions.png")


def plot_feature_distributions_log(X: pl.DataFrame, figsize=(20, 18)):
    feature_cols = [c for c in X.columns if c != "agent"]
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes      = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax     = axes[i]
        values = X[col].drop_nulls().to_numpy()
        values = values[np.isfinite(values)]

        # log transform for positive features
        if values.min() > 0:
            values = np.log1p(values)
            xlabel = f"log1p({col})"
        else:
            xlabel = col

        ax.hist(values, bins=50, edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("count")
        ax.axvline(np.median(values), color="red",    linestyle="--", linewidth=1, label=f"median={np.median(values):.2g}")
        ax.axvline(np.mean(values),   color="orange", linestyle="--", linewidth=1, label=f"mean={np.mean(values):.2g}")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions (log scale)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("output/feature_distributions_log.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/feature_distributions_log.png")