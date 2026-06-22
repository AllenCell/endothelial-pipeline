"""Helper function for visualizing spatial feature values on a crop grid."""

from pathlib import Path

import colorcet as cc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM


def create_panel_spatial_feature_grid(
    example_dataframes: list[pd.DataFrame],
    feature_columns: list[str],
    example_labels: list[str] | None = None,
    crop_size: int = 256,
    grid_start_xy: tuple[int, int] | None = None,
    grid_dimensions: tuple[int, int] = (3, 3),
    grid_spacing: int | None = None,
    start_x_col: str = "start_x",
    start_y_col: str = "start_y",
    cmap: str = "viridis",
    figure_size: tuple[float, float] | None = None,
    save_dir: Path | None = None,
    filename: str = "spatial_feature_grid",
) -> plt.Figure:
    """Create a figure showing spatial feature values on a grid for multiple examples.

    Rows correspond to feature columns, columns correspond to examples.
    Each panel shows a grid of colored patches where color encodes the
    feature value at that spatial crop position.

    Parameters
    ----------
    example_dataframes
        List of DataFrames, one per example. Each should be pre-filtered
        to a single position/timepoint.
    feature_columns
        List of column names to visualize (one per row).
    example_labels
        Optional labels for each example (used as column titles).
    crop_size
        Size of each crop in pixels (used for patch width/height).
    grid_start_xy
        Optional (start_x, start_y) of the upper-left crop in the grid.
        Combined with ``grid_dimensions`` and ``grid_spacing`` to compute
        which grid positions to include. If None, all positions present
        in the data are plotted.
    grid_dimensions
        (n_cols, n_rows) number of crops in the x and y directions.
        Only used when ``grid_start_xy`` is provided.
    grid_spacing
        Distance in pixels between adjacent crop start positions.
        Defaults to ``crop_size`` if not provided (i.e. non-overlapping grid).
        Set to a smaller value (e.g. 128) for overlapping crops on a
        finer grid.
    start_x_col
        Column name for the x-coordinate of each crop.
    start_y_col
        Column name for the y-coordinate of each crop.
    cmap
        Matplotlib colormap name.
    figure_size
        Total figure size (width, height) in inches. If None, auto-computed
        as 2.5 inches per example (columns) and 2.5 inches per feature (rows).
    save_dir
        Directory to save the output figure. If None, figure is not saved.
    filename
        Filename (without extension) for saving.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    # Compute grid positions from start_xy and dimensions
    spacing = grid_spacing if grid_spacing is not None else crop_size
    grid_positions: list[tuple[int, int]] | None = None
    if grid_start_xy is not None:
        n_cols_grid, n_rows_grid = grid_dimensions
        sx0, sy0 = grid_start_xy
        grid_positions = [
            (sx0 + col * spacing, sy0 + row * spacing)
            for row in range(n_rows_grid)
            for col in range(n_cols_grid)
        ]

    n_features = len(feature_columns)
    n_examples = len(example_dataframes)

    if figure_size is None:
        figure_size = (2.5 * n_examples, 2.5 * n_features)

    fig, axes = plt.subplots(
        n_features, n_examples, figsize=figure_size, squeeze=False, layout="constrained"
    )

    colormap_default = mpl_cm.get_cmap(cmap)

    for row_idx, feature in enumerate(feature_columns):
        # Determine colorbar scale from metadata
        metadata = COLUMN_METADATA.get(feature)
        if metadata is not None and metadata.min is not None and metadata.max is not None:
            vmin, vmax = float(metadata.min), float(metadata.max)
        else:
            vmin, vmax = None, None

        # Use circular colormap for polar angle, magenta/cyan for migration coherence
        if feature == Column.DiffAEData.POLAR_ANGLE:
            colormap = cc.cm.CET_C8
        elif feature == Column.OpticalFlow.UNIT_VECTOR_MEAN:
            colormap = mpl_cm.get_cmap("cool")
        else:
            colormap = colormap_default

        for col_idx, df_example in enumerate(example_dataframes):
            ax = axes[row_idx, col_idx]

            if feature not in df_example.columns:
                ax.set_visible(False)
                continue

            # Filter to specific grid positions if provided
            if grid_positions is not None:
                mask = pd.Series(False, index=df_example.index)
                for gx, gy in grid_positions:
                    mask |= (df_example[start_x_col] == gx) & (df_example[start_y_col] == gy)
                grid_data = df_example.loc[mask, [start_x_col, start_y_col, feature]].dropna(
                    subset=[feature]
                )
            else:
                grid_data = df_example[[start_x_col, start_y_col, feature]].dropna(subset=[feature])

            if grid_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                # Fall back to data range if metadata limits not available
                row_vmin = vmin if vmin is not None else grid_data[feature].min()
                row_vmax = vmax if vmax is not None else grid_data[feature].max()
                norm = Normalize(vmin=row_vmin, vmax=row_vmax)

                for _, data_row in grid_data.iterrows():
                    sx, sy = data_row[start_x_col], data_row[start_y_col]
                    val = data_row[feature]
                    color = colormap(norm(val))
                    rect = mpatches.FancyBboxPatch(
                        (sx, sy),
                        crop_size,
                        crop_size,
                        boxstyle="square,pad=0",
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

                # Set axis limits based on grid extent
                x_min = grid_data[start_x_col].min()
                x_max = grid_data[start_x_col].max() + crop_size
                y_min = grid_data[start_y_col].min()
                y_max = grid_data[start_y_col].max() + crop_size
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # invert y for image coords

            ax.set_aspect("equal")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Row labels (display names) on the left
            if col_idx == 0:
                label = metadata.label_with_unit if metadata is not None else feature
                ax.set_ylabel(label, fontsize=FONTSIZE_MEDIUM)

            # Column labels (example names) on top
            if row_idx == 0 and example_labels is not None:
                ax.set_title(example_labels[col_idx], fontsize=FONTSIZE_MEDIUM)

        # Add one colorbar per row on the far right
        row_vmin = vmin if vmin is not None else 0
        row_vmax = vmax if vmax is not None else 1
        norm = Normalize(vmin=row_vmin, vmax=row_vmax)
        sm = mpl_cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes[row_idx, :].tolist(), fraction=0.08, pad=0.04)
        cbar.outline.set_visible(False)
        if metadata is not None and metadata.ticks is not None:
            cbar.set_ticks(metadata.ticks)
            if metadata.tick_labels is not None:
                cbar.set_ticklabels(metadata.tick_labels)

    if save_dir is not None:
        save_plot_to_path(
            fig,
            save_dir,
            filename,
            file_format=".svg",
            tight_layout=False,
            pad_inches=0,
        )

    return fig
