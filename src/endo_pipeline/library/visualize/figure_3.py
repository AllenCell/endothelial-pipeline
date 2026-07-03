"""Helper functions for visualizations used in Figure 3."""

import logging
from pathlib import Path

import numpy as np

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.numerics.fixed_points import (
    load_fixed_points_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_vector_field_as_dict_from_dataframe,
    load_drift_dataframe_for_dataset,
)
from endo_pipeline.library.analyze.vector_field_function import get_callable_vector_field
from endo_pipeline.library.visualize.diffae_features.projected_dynamics import (
    plot_streamlines_of_projected_vector_field,
)
from endo_pipeline.library.visualize.figure_4 import wrap_theta_for_vector_field_vis
from endo_pipeline.library.visualize.figure_utils import set_axes_properties
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
from endo_pipeline.settings.flow_field_figure import XLABEL_KWARGS
from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


@figure_panel(
    "Make panel of 2D streamplot of projected vector field with stable fixed point overlay."
)
def visualize_2d_streamplots(
    dataset_name: str,
    output_path: Path,
    figure_size: tuple[float, float] = (2.0, 2.0),
) -> Path:
    # hard-coded parameters for visualization
    grid_spacing_2d = 0.05
    r_limits = (0.8, 1.6)
    theta_mesh, r_mesh = np.meshgrid(
        np.arange(VECTOR_FIELD_THETA_RANGE[0], VECTOR_FIELD_THETA_RANGE[1], grid_spacing_2d),
        np.arange(r_limits[0], r_limits[1], grid_spacing_2d),
    )
    theta_ticks = [0, np.pi / 2]
    theta_tick_labels = [f"0={Unicode.PI}", f"{Unicode.PI}/2"]
    r_ticks = [1.0, 1.5]
    theta_label = COLUMN_METADATA[Column.DiffAEData.POLAR_ANGLE].label or str(
        Column.DiffAEData.POLAR_ANGLE
    )
    r_label = COLUMN_METADATA[Column.DiffAEData.POLAR_RADIUS].label or str(
        Column.DiffAEData.POLAR_RADIUS
    )

    # load 3D data-driven vector field function and fixed points for dataset
    vector_field_dataframe = load_drift_dataframe_for_dataset(dataset_name)
    vector_field_dict = get_vector_field_as_dict_from_dataframe(
        vector_field_dataframe, column_names=list(DYNAMICS_COLUMN_NAMES)
    )
    vector_field_function = get_callable_vector_field(vector_field_dict, for_solve_ivp=False)

    fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)
    fixed_points_df = fixed_points_df[fixed_points_df[Column.FIXED_POINT_DETECTION_RATE] > 0.4]
    stable_df = fixed_points_df[
        fixed_points_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE
    ].copy()

    # modify theta coordinate to be within defined range used for 3D visualization
    stable_df.loc[:, Column.DiffAEData.POLAR_ANGLE] = stable_df[
        Column.DiffAEData.POLAR_ANGLE
    ].apply(
        lambda theta: wrap_theta_for_vector_field_vis(
            theta, VECTOR_FIELD_THETA_RANGE[0], VECTOR_FIELD_THETA_RANGE[1]
        )
    )

    # Change of basis to project onto plane defined by rho = rho_star: can
    # just take theta and r unit vectors as the new basis, since plane is
    # parallel to those axes.
    change_of_basis_matrix = np.eye(2, 3)
    stable_fixed_point_proj = stable_df[
        [Column.DiffAEData.POLAR_ANGLE, Column.DiffAEData.POLAR_RADIUS]
    ].to_numpy()[0]
    rho_star = float(stable_df[Column.DiffAEData.PC3_FLIPPED].iloc[0])

    fig = plot_streamlines_of_projected_vector_field(
        vector_field_function=vector_field_function,
        ortho_basis=change_of_basis_matrix,
        meshgrid_2d=(theta_mesh, r_mesh),
        figure_size=figure_size,
        fig_kwargs={"layout": "constrained"},
        streamplot_kwargs={"density": 0.8, "linewidth": 0.75, "color": "dimgrey"},
        origin_3d=np.array([0.0, 0.0, rho_star]),  # origin of projection in 3D space
    )

    # plot (theta, r) fixed point on top
    ax = fig.axes[0]
    ax.plot(
        stable_fixed_point_proj[0],
        stable_fixed_point_proj[1],
        FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
        color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=9,
        zorder=5,
    )

    # update theta ticks and set other axes properties
    set_axes_properties(
        ax,
        xlabel=theta_label,
        ylabel=r_label,
        xlabel_kwargs=XLABEL_KWARGS,
        ylabel_kwargs={"labelpad": 4, "rotation": 0},
        xticks=theta_ticks,
        xtick_labels=theta_tick_labels,
        yticks=r_ticks,
    )

    return save_plot_to_path(
        fig,
        output_path,
        figure_name=f"{dataset_name}_projected_streamplot",
        file_format=".svg",
        tight_layout=False,
    )
