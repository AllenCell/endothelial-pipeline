"""Helper functions for visualizations used in Figure 4."""

from pathlib import Path
from typing import cast

import numpy as np
from matplotlib import patheffects
from pandas import DataFrame

from endo_pipeline.io import load_dataframe, load_model, save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import load_drift_dataframe_for_dataset
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.diffae_features.dynamics import (
    plot_drift_3d,
    process_3d_vector_field_for_visualization,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES, POLAR_ANGLE_PERIOD
from endo_pipeline.settings.figures import FONTSIZE_XSMALL
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
)


def wrap_theta_for_vector_field_vis(theta, theta_min, theta_max, period=POLAR_ANGLE_PERIOD):
    if theta < theta_min:
        return theta + period
    elif theta > theta_max:
        return theta - period
    else:
        return theta


@figure_panel("Make panel of 3D vector field plot with stable fixed point overlay.")
def make_3d_vector_field_plot_panel(
    dataset_name: str,
    output_path: Path,
    figure_size: tuple[float, float] = (2.0, 2.5),
    include_colorbar: bool = True,
    include_legend: bool = True,
) -> tuple[Path, DataFrame]:
    """
    Render the 3D (theta, r, rho) drift vector field for a given dataset, with
    the stable fixed point(s) overlaid as a scatter marker.

    Parameters
    ----------
    dataset_name
        Name of the dataset to visualize.
    output_path
        Directory in which to save the figure.
    figure_size
        Size of the figure to create.
    include_colorbar
        Whether to include a colorbar indicating the magnitude of the drift
        vector.
    include_legend
        Whether to include a legend (vector arrow and fixed point marker).

    Returns
    -------
    :
        Path to the saved figure file and a DataFrame containing the
        coordinates of the stable fixed points that were plotted.

    """
    drift_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    feature_dataframe_manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
    feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])

    column_names = list(DYNAMICS_COLUMN_NAMES)  # [theta, r, rho]
    col_labels = [(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]
    fixed_point_label = f"({col_labels[0]}$^*$, {col_labels[1]}$^*$, {col_labels[2]}$^*$)"

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
        figsize=figure_size,
        include_colorbar=include_colorbar,
        include_legend=include_legend,
        fixed_point_legend_label=fixed_point_label,
        colorbar_rect=(0.5, 0.12, 0.4, 0.02),
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
        xlabel_kwargs={"labelpad": -8},
        ylabel_kwargs={"labelpad": -5},
        zlabel_kwargs={"labelpad": -8},
    )

    # Load and overlay stable fixed point
    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    fixed_points_df = fixed_points_df[fixed_points_df[Column.FIXED_POINT_DETECTION_RATE] > 0.4]
    stable_df = fixed_points_df[
        fixed_points_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE
    ]
    color: str = FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color
    column_names_ = cast(list[str], column_names)
    for _, fpt_row in stable_df.iterrows():
        fpt_coords = fpt_row[column_names_].to_numpy()
        # wrap theta coordinate to be within the specified limits
        fpt_coords[0] = wrap_theta_for_vector_field_vis(fpt_coords[0], theta_lims[0], theta_lims[1])
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
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
    )

    return output_path / f"{filename}.svg", stable_df


@figure_panel("Reconstruct at fixed points.")
def reconstruct_fixed_points(
    fixed_point_df: DataFrame,
    output_path: Path,
    figure_size: tuple[float, float] = (0.8, 1.6),
    num_gpus: int | None = None,
    random_seed: int | None = 4,
    add_fixed_point_coordinate_annotation: bool = False,
) -> Path:
    """
    Reconstruct the fixed point coordinates from the polar angle, radius, and
    rho columns.

    Parameters
    ----------
    fixed_point_df
        DataFrame containing the fixed point coordinates.
    output_path
        Directory to save the reconstructed images.
    figure_size
        Size of the figure for the contact sheet of images.
    num_gpus
        Number of GPUs to use for reconstruction, by default None
    random_seed
        Random seed for reproducibility, by default 4
    add_fixed_point_coordinate_annotation
        Whether to add fixed point coordinate annotations to the reconstructed
        images, by default False.
    """

    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    column_names = cast(list[str], list(DYNAMICS_COLUMN_NAMES))
    column_labels = [str(COLUMN_METADATA[col].label or str(col)) for col in DYNAMICS_COLUMN_NAMES]

    # wrap the polar angle to be within the specified limits for visualization
    # and consistency with the vector field plots
    fixed_point_df[Column.DiffAEData.POLAR_ANGLE] = (
        fixed_point_df[Column.DiffAEData.POLAR_ANGLE] % POLAR_ANGLE_PERIOD
    )

    theta_min, theta_max = VECTOR_FIELD_THETA_RANGE
    fixed_point_df[Column.DiffAEData.POLAR_ANGLE] = fixed_point_df[
        Column.DiffAEData.POLAR_ANGLE
    ].transform(
        lambda theta, theta_min=theta_min, theta_max=theta_max: wrap_theta_for_vector_field_vis(
            theta, theta_min, theta_max
        )
    )

    # sort the fixed points dataframe by polar angle so that the reconstructions
    # are in a consistent and predictable order
    fixed_point_df = fixed_point_df.sort_values(by=Column.DiffAEData.POLAR_ANGLE)

    # reconstruct images along at the fixed point coordinates and make a contact
    # sheet of the results
    walk_array = generate_from_dataframe(
        fixed_point_df,
        column_names,
        model,
        num_gpus=num_gpus,
        random_seed=random_seed,
    )
    walk_panels = [walk_array[i] for i in range(len(walk_array))]

    fig_fixed_point_reconstructions = make_contact_sheet(
        panels=walk_panels,
        max_rows=2,
        max_cols=1,
        fig_kwargs={"figsize": figure_size, "layout": "constrained"},
        gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
    )

    # add the fixed point coordinate as an annotation to each panel
    if add_fixed_point_coordinate_annotation:
        for i, ax in enumerate(fig_fixed_point_reconstructions.axes):
            fp_coords = fixed_point_df.iloc[i][column_names].to_numpy(dtype=float)
            fp_coords_list = [
                f"{col} = {fp_coords[j].round(2)}" for j, col in enumerate(column_labels)
            ]
            fp_coords_as_str = "\n".join(fp_coords_list)

            fp_annotation = ax.text(
                x=0.02,
                y=0.98,
                s=fp_coords_as_str,
                ha="left",
                va="top",
                fontsize=FONTSIZE_XSMALL,
                color="white",
                transform=ax.transAxes,
            )
            fp_annotation.set_path_effects(
                [patheffects.withStroke(linewidth=0.5, foreground="black")]
            )

    # add scalebars to each panel, only label the top left one to avoid
    # redundancy
    for i, ax in enumerate(fig_fixed_point_reconstructions.axes):
        add_scalebar(
            ax,
            scale_bar_um=20,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            location="lower right",
            bar_thickness=5,
            padding=5,
            include_label=True if i == 0 else False,
            label_fontsize=FONTSIZE_XSMALL,
        )

    dataset_name = fixed_point_df[Column.DATASET].unique()[0]
    filename = f"{dataset_name}_fixed_point_reconstructions"
    save_plot_to_path(
        fig_fixed_point_reconstructions,
        output_path,
        filename,
        file_format=".svg",
        tight_layout=False,
        transparent=True,
        bbox_inches="tight",
    )

    return output_path / f"{filename}.svg"
