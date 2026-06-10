"""Helper functions for visualizations used in Figure 3."""

from pathlib import Path

import numpy as np

from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import load_drift_dataframe_for_dataset
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    plot_drift_3d,
    process_3d_vector_field_for_visualization,
)
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME


@figure_panel("Make panel of 3D vector field plot with stable fixed point overlay.")
def make_3d_vector_field_plot_panel(
    dataset_name: str,
    fig_savedir: Path,
) -> Path:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset, with
    the stable fixed point overlaid as a scatter marker.

    Parameters
    ----------
    dataset_name
        Name of the dataset to visualize.
    fig_savedir
        Directory in which to save the figure as a static PNG file.

    Returns
    -------
    :
        Path to the saved figure file.

    """
    drift_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    feature_dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]

    theta_lims = VECTOR_FIELD_THETA_RANGE
    r_lims = (0, 1.75)
    rho_lims = (-1.5, 1.5)

    # Load, clip, and downsample drift vector field
    drift, meshgrid = process_3d_vector_field_for_visualization(
        drift_dataframe,
        feature_dataframe,
        column_names=column_names,
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        mask_threshold=0.025,
    )

    fig, ax = plot_drift_3d(
        drift=drift,
        meshgrid=meshgrid,
        figsize=(2.0, 2.5),
        xlim=theta_lims,
        ylim=r_lims,
        zlim=rho_lims,
        xticks=[0, np.pi / 2],
        xtick_labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"],
        yticks=[0.25, 0.75, 1.25],
        zticks=[-1.0, 0, 1.0],
        xlabel=col_labels[0],
        ylabel=col_labels[1],
        zlabel=col_labels[2],
    )

    # Load and overlay stable fixed point
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

    stable_df = fixed_points_df[
        fixed_points_df[Column.VectorField.STABILITY] == StabilityLabel.STABLE
    ]
    color: str = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color
    for _, fpt_row in stable_df.iterrows():
        fpt_coords = fpt_row[column_names].to_numpy()
        ax.scatter(
            fpt_coords[0],
            fpt_coords[1],
            fpt_coords[2],
            color=color,
            s=15,
            zorder=5,
        )

    # save as .svg file
    filename = f"3d_vector_field_{dataset_name}"
    save_plot_to_path(
        fig,
        fig_savedir,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=False,
        bbox_inches="tight",
    )

    return fig_savedir / f"{filename}.svg"
