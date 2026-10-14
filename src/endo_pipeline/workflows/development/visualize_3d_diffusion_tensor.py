from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets, PatchType


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    use_same_axes: Annotated[bool, Parameter(negative="--use-auto-axes")] = False,
) -> None:
    """
    Visualize entries of the diffusion tensor over 2D slices of 3D state space.

    #dynamical-systems #grid-based #cell-centered #visualization #test-ready

    This workflow uses the precomputed noise amplitude tensor output by the
    `compute-noise-amplitude` workflow, run for all three dynamics column
    names. Make sure to run that workflow with the matching patch type before
    visualizing.

    The noise amplitude `sigma` solves `D = (1/2) sigma sigma^T` for the
    symmetric Kramers-Moyal diffusion tensor `D` (see `compute_noise_amplitude`
    in `sde_interpretation.py`). This workflow visualizes the unique entries
    `sigma_ij(x)` directly, as contour plots over 2D slices of the 3D state
    space.

    Visualization outputs include:

    - For each of the three possible slicing axes, a stack of contour plots
      (one per grid value along that axis) showing every unique entry of the
      noise amplitude tensor over the corresponding 2D slice of state space.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-3d-diffusion-tensor -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-3d-diffusion-tensor --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize
    the diffusion tensor for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to visualize.
    use_same_axes
        True to use global limits across all datasets, False otherwise.
    """

    import logging
    from itertools import combinations_with_replacement

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_shear_stress
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.sde_interpretation import (
        extrapolate_noise_amplitude_tensor,
        get_reshaped_noise_amplitude_and_grid,
    )
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.flow_field_3d import (
        plot_diffusion_tensor_stack,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
    )
    from endo_pipeline.settings.manifest_names import NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__, patch_type)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Use all three dynamics columns, matching the compute-noise-amplitude output
    column_names = list(DYNAMICS_COLUMN_NAMES)
    column_labels = [get_label_for_column(column) for column in column_names]
    mesh_column_names = [ColumnTemplate.MESH_GRID % column for column in column_names]
    num_dimensions = len(column_names)

    # Unique (i, j) entries and their panel titles/column names, labeled with i <= j
    entry_pairs = list(combinations_with_replacement(range(num_dimensions), 2))
    entry_labels = [
        f"{Unicode.SIGMA}({column_labels[i]}, {column_labels[j]})" for i, j in entry_pairs
    ]
    entry_column_names = [
        ColumnTemplate.NOISE_AMPLITUDE % (column_names[i], column_names[j]) for i, j in entry_pairs
    ]
    # sigma is the lower-triangular Cholesky factor, so the non-trivial entry
    # for each (i, j) pair with i < j lives at tensor index (j, i), not (i, j).
    tensor_index_pairs = [(j, i) for i, j in entry_pairs]
    required_columns = [
        *mesh_column_names,
        *entry_column_names,
        Column.DATASET,
        Column.SHEAR_STRESS,
    ]

    bin_widths = [BIN_WIDTHS_DYNAMICS[column] for column in column_names]
    bounds_for_plots = [BIN_LIMITS_DYNAMICS[column] for column in column_names]

    # Load precomputed noise amplitude tensor for selected columns
    name_suffix = join_sorted_strings(column_names)
    noise_amplitude_manifest_name = (
        f"{NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
    )
    noise_amplitude_manifest = load_dataframe_manifest(noise_amplitude_manifest_name)

    # (axes to plot in 2D, axis being sliced over) for each of the three slice stacks
    plot_axes_indices = [(0, 1), (0, 2), (1, 2)]
    slice_axis_indices = [2, 1, 0]

    for dataset_name in dataset_names:
        if dataset_name not in noise_amplitude_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                noise_amplitude_manifest_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)

        noise_amplitude_dataframe_location = get_dataframe_location_for_dataset(
            noise_amplitude_manifest, dataset_name
        )
        noise_amplitude_dataframe = load_dataframe(noise_amplitude_dataframe_location, delay=False)
        check_required_columns_in_dataframe(noise_amplitude_dataframe, required_columns)

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"
            fig_title = get_shear_stress_label_for_dataset(dataset_config, flow_condition)

            noise_amplitude_for_flow_condition = filter_dataframe_by_shear_stress(
                noise_amplitude_dataframe, shear_stress
            )
            if noise_amplitude_for_flow_condition.empty:
                logger.warning(
                    "No noise amplitude tensor found for [ %s ]. Skipping.", dataset_name_flow
                )
                continue

            sigma, grid_points_1d = get_reshaped_noise_amplitude_and_grid(
                noise_amplitude_for_flow_condition, column_names
            )
            # fill NaN bins (too few samples) before plotting, so sparse
            # per-bin estimates don't leave slices entirely blank
            sigma = extrapolate_noise_amplitude_tensor(sigma, grid_points_1d)
            grid = np.meshgrid(*grid_points_1d, indexing="ij")

            # if not using same axes for all datasets, use bin limits for this
            # specific dataset, to zoom in on the region covered by this grid
            if not use_same_axes:
                bounds_for_plots = [
                    (
                        grid_points_1d[i][0] - bin_widths[i] / 2,
                        grid_points_1d[i][-1] + bin_widths[i] / 2,
                    )
                    for i in range(num_dimensions)
                ]

            for plot_axes, slice_axis in zip(plot_axes_indices, slice_axis_indices, strict=True):
                slice_column_name = column_names[slice_axis]
                logger.info(
                    "Plotting diffusion tensor contours for [ %s ], slice axis [ %s ].",
                    dataset_name_flow,
                    slice_column_name,
                )
                stack_savedir = output_path / f"{dataset_name_flow}_{slice_column_name}_stack"
                stack_savedir.mkdir(parents=True, exist_ok=True)
                plot_diffusion_tensor_stack(
                    sigma,
                    grid,
                    plot_axes_indices=plot_axes,
                    slice_axis_index=slice_axis,
                    plot_bounds=bounds_for_plots,
                    slice_steps=grid_points_1d[slice_axis],
                    entry_pairs=tensor_index_pairs,
                    entry_labels=entry_labels,
                    fig_savedir=stack_savedir,
                    feature_labels=column_labels,
                    fig_title=fig_title,
                )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
