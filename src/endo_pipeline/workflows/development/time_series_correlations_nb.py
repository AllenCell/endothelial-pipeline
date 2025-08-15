# %%
DESCRIPTION = "Run auto- and cross-correlation analysis on DiffAE feature time series data."

TAGS = ["diffae_features"]

import logging
import sys
from typing import cast

import numpy as np

from src.endo_pipeline.configs import CytoDLModelConfig, get_model_manifest, load_model_config
from src.endo_pipeline.io import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest import (
    df_to_array,
    fit_pca,
    get_manifest_for_dynamics_workflows,
    get_pc_column_names,
)
from src.endo_pipeline.library.analyze.numerics import (
    autocorrelation_function,
    cross_correlation_function,
    exponential_decay,
    fit_exponential_decay,
)
from src.endo_pipeline.library.visualize import viz_base

# if being run as a script, use logging
# instead of print statements
print_statements = False
logger = logging.getLogger(__name__)

# set up print statements instead of logging
# if this script is being run as a notebook
if hasattr(sys, "ps1"):
    print_statements = True


# %%
# fit PCA object
model_name = "diffae_04_10"
pca = fit_pca(model_name=model_name)

# list of datasets to process
list_of_datasets = [
    "20250409_20X",
    "20250319_20X",
    "20250326_20X",
    "20241120_20X",
]

# load model config to get model manifest objects
model_config = cast(CytoDLModelConfig, load_model_config(model_name))

figure_save_path = get_output_path("correlations")

# %%
for dataset_name in list_of_datasets:
    logger.info("Processing dataset [ %s ] for correlation analysis", dataset_name)
    # load dataframe and get top 3 PCs
    model_manifest = get_model_manifest(dataset_name, model_config)
    df = get_manifest_for_dynamics_workflows(model_manifest, pca)
    feat_cols = get_pc_column_names(df, pc_axes=[0, 1, 2])

    # get feature data
    feats = df_to_array(df, feat_cols)

    num_timepoints = feats.shape[1]
    # make sure lags are symmetric around zero
    if num_timepoints % 2 == 0:
        # even number of timepoints
        lags = np.arange(-num_timepoints // 4 + 1, num_timepoints // 4)
    else:
        # odd number of timepoints
        lags = np.arange(-num_timepoints // 4 + 2, num_timepoints // 4)

    num_lags = len(lags)
    # autocorrelation
    acf = np.zeros((num_lags, 3))
    for i in range(3):
        for k in range(num_lags):
            acf[k, i] = autocorrelation_function(feats, i, lags[k])

    ccf = np.zeros((num_lags, 3))
    index_combinations = [(0, 1), (0, 2), (1, 2)]
    for i, (j, k) in enumerate(index_combinations):
        for lag_index in range(num_lags):
            ccf[lag_index, i] = cross_correlation_function(feats, j, k, lags[lag_index])

    # get difference between
    # forward and backward lags
    # leave out zero
    delta_ccf = np.zeros((num_lags // 2, 3))
    for i, _ in enumerate(index_combinations):
        delta_ccf[:, i] = ccf[1 + num_lags // 2 :, i] - ccf[: num_lags // 2, i]

    # plot acf for positive lags
    # (acf is symmetric around zero)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    index_positive = lags > 0
    lags_ = lags[lags > 0]
    acf_ = acf[lags > 0]
    fig, ax = viz_base.init_plot(figsize=(12, 6))
    for i in range(3):
        ax.plot(lags_, acf_[:, i], "k", linewidth=3.0, label="")
        ax.plot(lags_, acf_[:, i], linewidth=2.75, label=f"PC{i+1}")

    ax.set_title(f"Autocorrelation of PCA Components ({dataset_name})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    save_plot_to_path(
        fig,
        figure_save_path,
        f"autocorrelation_{dataset_name}",
    )

    # fit exponential decay to ACF
    for i in range(3):
        exp_fit = fit_exponential_decay(lags_, acf_[:, i])
        relaxation_time = 5 * (1 / exp_fit[1]) / 60  # convert to hours
        if print_statements:
            print(
                f"PC {i + 1} relaxation timescale: {relaxation_time:.2f} hrs.",
            )
        logger.info(
            "PC %d relaxation timescale: %.2f hrs.",
            i + 1,
            relaxation_time,
        )
        acf_fit = exponential_decay(lags_, *exp_fit)
        ax.plot(lags_, acf_fit, "k--", linewidth=2.0, alpha=0.85, label="")
    ax.legend()
    ax.set_ylim(-0.25, 1.05)
    save_plot_to_path(
        fig,
        figure_save_path,
        f"autocorrelation_fit_{dataset_name}",
    )

    # plot ccf
    fig, ax = viz_base.init_plot(figsize=(12, 6))
    for i, (j, k) in enumerate(index_combinations):
        ax.plot(lags, ccf[:, i], label=f"(PC{j+1}, PC{k+1})")

    ax.set_title(f"Cross-Correlation of PCA Components ({dataset_name})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("CCF")
    ax.legend()
    ax.set_ylim(-0.25, 0.75)
    save_plot_to_path(
        fig,
        figure_save_path,
        f"cross_correlation_{dataset_name}",
    )

    # plot delta ccf
    fig, ax = viz_base.init_plot(figsize=(12, 6))
    for i, (j, k) in enumerate(index_combinations):
        ax.plot(lags[1 + num_lags // 2 :], delta_ccf[:, i], label=f"(PC{j+1}, PC{k+1})")
    ax.set_title("$C_{ij}(\\tau) - C_{ij}(-\\tau)$" + f" ({dataset_name})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("$\Delta C_{ij}(\\tau)$")
    ax.legend()
    ax.set_ylim(-0.75, 0.75)
    save_plot_to_path(
        fig,
        figure_save_path,
        f"cross_correlation_diff_{dataset_name}",
    )
    # log summary statistics
    if print_statements:
        print(
            f"Minimum, maximum, and mean of delta CCF for dataset [ {dataset_name} ]: "
            f"[ {np.min(delta_ccf, axis=0)}, {np.max(delta_ccf, axis=0)}, "
            f"{np.mean(delta_ccf, axis=0)} ] \n \n"
        )
    logger.info(
        "Minimum, maximum, and mean of delta CCF for dataset [ %s ]:" " [ %s, %s, %s ]",
        dataset_name,
        np.min(delta_ccf, axis=0),
        np.max(delta_ccf, axis=0),
        np.mean(delta_ccf, axis=0),
    )
# %%
