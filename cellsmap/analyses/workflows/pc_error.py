# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from typing import Any
from numpy.typing import ArrayLike
from pathlib import Path

"""
This workflow is used to calulcate the error in the PCA values for several datasets,
when the PCs from one experiment to a related experiment (for example, pre-fixation to post-fixation).
The error is calculated for any single PC for any dataset by:
    1. Plotting the PC values for the two experiments against each other
    2. Fitting a confidence ellipse to the set of PC values from the paired experiments
    3. Projecting the height of the confidence ellipse onto the y-axis to get an error value
Running `main` will run the analysis on all datasets currently designated for this analysis
in the `datasets` dictionary as a standalone workflow to generate figures demoing how PC error
is calculated for each PC and each dataset.
To just extract a error measurement for paired data to draw error bars in a figure,
the `pc_error_from_confidence_ellipse` function can be used directly. It currently just takes in 
x and y data from the two experiments, and can output the resulting error value. We can fine tune
how the data passing works when the data used here is actually on FMS and being passed into
different analyses.

Reference: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
"""


# %% Load data

# This creates a dictionary of datasets: each dataset is itself a pair of experiments,
# defined as dictionary including the data for each of two experiments,
# and (for plotting purposes) human-readable names for the dataset and labels for each experiment.
basedir = "/allen/aics/users/benjamin.morris/cellsmap/results/models/diffae_04_10"
datasets: dict = {
    "20250214_pairedPrePostFixation_diffae_04_10_features": {
        "name": "20250214 Paired Pre/Post Fixation",
        "x": f"{basedir}/20250214_pairedPreFixation_vs_20250214_pairedPostFixation/predict_20250214_pairedPreFixation_diffae_04_10_features.parquet",
        "y": f"{basedir}/20250214_pairedPreFixation_vs_20250214_pairedPostFixation/predict_20250214_pairedPostFixation_diffae_04_10_features.parquet",
        "xlabel"    : "20x 3i pre-fixation",
        "ylabel"    : "20x 3i post-fixation",
    },
    "20250227_20x_hrmanual_diffae_04_10_features": {
        "name": "20250227 20x/40x 3i live",
        "x": f"{basedir}/20250227_20x_vs_20250227_hr_manual/predict_20250227_20x_diffae_04_10_features.parquet",
        "y": f"{basedir}/20250227_20x_vs_20250227_hr_manual/predict_20250227_hr_manual_diffae_04_10_features.parquet",
        "xlabel"    : "20x 3i live",
        "ylabel"    : "40x 3i live",
    },
    "20250110_20x_hrmanual_diffae_04_10_features": {
            "name": "20250110 20x/40x 3i live",
            "x": f"{basedir}/20250110_20x_vs_20250110_hr_manual/predict_20250110_20x_diffae_04_10_features.parquet",
            "y": f"{basedir}/20250110_20x_vs_20250110_hr_manual/predict_20250110_hr_manual_diffae_04_10_features.parquet",
            "xlabel"    : "20x 3i live",
            "ylabel"    : "40x 3i live",
        },
}

# %% Define functions

def pc_error_from_confidence_ellipse(x: ArrayLike, y: ArrayLike, ax: plt.Axes, n_std: float=3.0, facecolor: str ='none', **kwargs: Any)-> tuple[plt.Axes, Ellipse, float]:
    """
    Create a plot of the covariance confidence ellipse of *x* and *y* and caluclate the
    associated PC error (the y-axis projection of the ellipse height.
    This function is adapted from the code in the reference link in the header of this file.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data from paired experiments.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x: float = np.mean(x)

    # Calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y: float = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    yerr = ellipse.height * scale_y *np.sin(45)
    ax.add_patch(ellipse)
    ellipse.set_label(rf'${n_std}\sigma$, y_err: {yerr:.2f}')
    return ax, ellipse, yerr


def plot_ellipse123std(x: ArrayLike, y: ArrayLike, pc: int, dataset: str, name: str, xlabel: str, ylabel: str) -> None:
    """
    Create a scatter plot of the paired *pc* data given as *x* and *y* from a given
    *dataset* with a humanreadable *name*. Then calculate and overlay covariance confidence
    ellipses of *x* and *y* for 1, 2 and 3 standard deviations out from the mean. The 
    Includes a y=x line for reference.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data from paired experiments.

    pc : int
        PC to analyze.

    dataset : str
        Dataset name formatted for code and filepath handling.

    name : str
        Human-readable name for the dataset.
    
    xlabel : str
        Human-readable label for experiment whose PC data is plotted on the x-axis.

    ylabel : str
        Human-readable label for experiment whose PC data is plotted on the y-axis.
    """

    # Create scatter plot of PC data from two experiments
    ax = plt.gca()
    ax.scatter(x, y, s=0.5, c='black', alpha=0.2)
    # Add a reference line for y=x where PCs for the two experiments are equal
    ax.axline((0, 0), slope=1, linestyle='--', c='black', lw=1)

    # Calculate and plot confidence ellipses for 1, 2 and 3 standard deviations
    # and project the height of the ellipses onto the y-axis to get an error value
    # for each ellipse which is provided in the figure legend
    for n_std, c in zip([1, 2, 3], "firebrick fuchsia blue".split()):
        pc_error_from_confidence_ellipse(x, y, ax, n_std=n_std, edgecolor=c)

    # Format, label, and save the figure
    plt.legend(loc='upper left')
    plt.xlabel(f"PC{pc} {xlabel}")
    plt.ylabel(f"PC{pc} {ylabel}")
    plt.title(f"PC{pc}, Dataset: {name}")
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    prj_dir = Path(__file__).parent.parent.parent.parent
    savedir = prj_dir / "results" / "pc_error_data_integration"
    savedir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{savedir}/{dataset}_PC{pc}.png",dpi=300,)


def run_pc_error_on_single_dataset(dataset: str, PC_list:list = [1, 2, 3]) -> None:
    """
    Run PC error analysis on the given dataset for the given set of PCs in *PC_list*.
    """

    # load the data for the paired experiment and extract labels from the dataset dictionary
    if dataset not in datasets.keys():
        raise ValueError(f"Dataset {dataset} not found in datasets dictionary")
    else:
        dataset_dict = datasets[dataset]
        pre_df = pd.read_parquet(dataset_dict["x"])
        post_df = pd.read_parquet(dataset_dict["y"])
        name = dataset_dict["name"]
        xlabel = dataset_dict["xlabel"]
        ylabel = dataset_dict["ylabel"]

        # loop through the PCs in the list and plot the confidence ellipse and calculate
        # the error for each PC to include in the plot legend
        for pc in PC_list:
            plt.clf()
            if f"pc{pc}" not in pre_df.columns:
                print(f"PC{pc} not found in dataset {dataset}")
            else:
                x, y = pre_df[f"pc{pc}"].values, post_df[f"pc{pc}"].values
                plot_ellipse123std(x, y, pc, dataset, name, xlabel, ylabel)

def run_pc_error_on_all_datasets() -> None:
    """
    Main workflow function.
    Run PC error analysis on all datasets in the datasets dictionary.
    Apply for PCs 1, 2, and 3.
    """
    for dataset in datasets.keys():
        run_pc_error_on_single_dataset(dataset)


# %% Plot
if __name__ == '__main__':
    run_pc_error_on_all_datasets()

