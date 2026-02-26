import matplotlib.pyplot as plt
import numpy as np


def plot_lda_vs_optical_flow(
    df,
    features,
    optical_flow_features,
    color_by_dataset=True,
    point_alpha=0.25,
    figsize=(24, 2.5 * 9),
    max_points=10000,
):
    # Drop rows with NaNs in relevant columns
    df = df.dropna(subset=features + optical_flow_features + ["dataset"])
    rng = np.random.default_rng(42)
    datasets_used = df["dataset"].unique()
    n_rows, n_cols = len(optical_flow_features), len(features)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex="col", sharey="row")
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape((n_rows, n_cols))

    if color_by_dataset:
        colors = plt.cm.tab10(np.arange(len(datasets_used)))
    else:
        colors = ["tab:blue"] * len(datasets_used)

    df_np = df.copy()
    for col in features + optical_flow_features + ["dataset"]:
        if col not in df_np:
            continue
        df_np[col] = df_np[col].to_numpy()

    # Compute N for each dataset for legend
    legend_labels = []
    for i, dataset in enumerate(datasets_used):
        mask = df["dataset"] == dataset
        n_points = np.sum(mask)
        legend_labels.append(f"{dataset} (N={n_points})")

    for row, of_feature in enumerate(optical_flow_features):
        of_data = df_np[of_feature]
        for col, feature in enumerate(features):
            ax = axes[row, col]
            feat_data = df_np[feature]

            # calculate correlation coefficient
            corr_coef = np.corrcoef(feat_data, of_data)[0, 1]

            for i, dataset in enumerate(datasets_used):
                mask = df["dataset"] == dataset
                x_full = feat_data[mask]
                y_full = of_data[mask]
                x, y = x_full, y_full
                if max_points is not None and len(x) > max_points:
                    idx = rng.choice(len(x), max_points, replace=False)
                    x = x.iloc[idx]
                    y = y.iloc[idx]
                ax.scatter(
                    x,
                    y,
                    alpha=point_alpha,
                    color=colors[i],
                    label=legend_labels[i] if (row == 0 and col == 0) else None,  # Only label once
                    rasterized=True,
                )
            if row == 0:
                ax.set_title(feature)
            if col == 0:
                ax.set_ylabel(of_feature)
            ax.annotate(f"r={corr_coef:.2f}", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=10)
            ax.grid(True)

    # Add legend to the first axis with handles
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=10, frameon=True)
            break

    plt.tight_layout()
    plt.show()
    plt.close(fig)
