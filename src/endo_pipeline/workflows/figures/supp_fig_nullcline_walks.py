from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Compile panels visualizing nullcline walks.

    - **Panel A*: 2D contour plot of drift coefficients for polar radius and rho
      (-PC3) coordinates for example low shear stress dataset.
    - **Panel B**: DiffAE generated synthetic images along nullcline in polar
      radius and rho (-PC3) coordinates for example low shear stress dataset.
    - **Panels C-D**: Same as panels A-B but for example high shear stress dataset.

    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, load_model
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
        load_drift_dataframe_for_dataset,
    )
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.nullcline_walks import (
        make_contour_plot_panel_for_nullcline_walks,
        reconstruct_along_nullcline,
    )
    from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameSuffix
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D"])

    # figure is for grid based crops
    crop_pattern = "grid"

    dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
    dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]

    columns_r_rho = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    columns_r_rho_fixed_point = [f"{col}{ColumnNameSuffix.FIXED_POINTS}" for col in columns_r_rho]
    columns_r_rho_str = join_sorted_strings(columns_r_rho)
    column_theta = Column.DiffAEData.POLAR_ANGLE
    column_theta_fixed_point = f"{column_theta}{ColumnNameSuffix.FIXED_POINTS}"
    feature_column_names = [column_theta, *columns_r_rho]
    feature_columns_str = join_sorted_strings(feature_column_names)

    # load dataframe manifests for diffae features, fixed points, optical flow
    # features, and bootstrapped fixed points for this crop pattern, which will be
    # used for all visualizations in this figure
    name_suffix_2d = f"_{columns_r_rho_str}_{crop_pattern}"
    fixed_points_r_rho_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix_2d}"
    )
    fixed_points_r_rho_dataframe_manifest = load_dataframe_manifest(
        fixed_points_r_rho_dataframe_manifest_name
    )
    name_suffix_1d = f"_{column_theta}_{crop_pattern}"
    fixed_points_theta_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix_1d}"
    )
    fixed_points_theta_dataframe_manifest = load_dataframe_manifest(
        fixed_points_theta_dataframe_manifest_name
    )
    name_suffix_3d = f"_{feature_columns_str}_{crop_pattern}"
    fixed_points_3d_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix_3d}"
    )

    fixed_points_3d_dataframe_manifest = load_dataframe_manifest(
        fixed_points_3d_dataframe_manifest_name
    )

    # get labels for provided set of feature columns
    column_labels_r_rho = [COLUMN_METADATA[column].label for column in columns_r_rho]

    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
    contour_plot_paths: dict[str, Path] = {}
    nullcline_reconstruction_paths: dict[str, Path] = {}
    for dataset_name in [dataset_low, dataset_high]:
        fig_savedir = get_output_path(__file__, dataset_name)

        # load fixed points dataframes (if available) for both (r, rho) and theta,
        # filter to just stable fixed points, and store in dict for easy access when plotting
        stable_fixed_points_dict: dict[
            tuple[Column.DiffAEData] | Column.DiffAEData, pd.DataFrame | None
        ] = {}
        for column_key, manifest in [
            (columns_r_rho_str, fixed_points_r_rho_dataframe_manifest),
            (column_theta, fixed_points_theta_dataframe_manifest),
            (feature_columns_str, fixed_points_3d_dataframe_manifest),
        ]:
            df_fixed_points = load_dataframe(manifest.locations[dataset_name])
            df_stable_fixed_points = filter_dataframe_by_stability(
                df_fixed_points, stability_label=StabilityLabel.STABLE
            )
            stable_fixed_points_dict[column_key] = df_stable_fixed_points

        drift_r_rho_dataframe = load_drift_dataframe_for_dataset(
            dataset_name, columns=columns_r_rho
        )
        drift_r_rho, centers_r_rho = get_reshaped_vector_field_and_grid(
            drift_r_rho_dataframe,
            column_names=columns_r_rho,
        )
        centers_mesh = np.meshgrid(*centers_r_rho, indexing="ij")
        stable_fixed_point_r_rho = stable_fixed_points_dict[columns_r_rho_str][
            columns_r_rho_fixed_point
        ].to_numpy()

        stable_fixed_point_theta = stable_fixed_points_dict[column_theta][
            column_theta_fixed_point
        ].to_numpy()

        contour_plot_paths[dataset_name], nullcline_coordinates = (
            make_contour_plot_panel_for_nullcline_walks(
                figure_size=(2.6, 1.55),
                output_path=fig_savedir,
                drift=drift_r_rho,
                meshgrid=centers_mesh,
                column_labels=column_labels_r_rho,
                stable_fixed_point=stable_fixed_point_r_rho,
                filename=f"{dataset_name}_{columns_r_rho_str}_contours",
                plot_nullcline_walk_points=True,
                **placeholders["B"],
            )
        )

        nullcline_reconstruction_paths[dataset_name] = reconstruct_along_nullcline(
            figure_size=(3.125, 1.3),
            output_path=fig_savedir,
            nullcline_coords=nullcline_coordinates,
            theta_value=stable_fixed_point_theta[0],
            model=model,
            num_gpus=NUM_GPUS,
            random_seed=4,
            **placeholders["C"],
        )

    # --- Assemble all panels into final figure ---
    panels = [
        FigurePanel(
            letter="A",
            path=contour_plot_paths[dataset_low],
            x_position=0.0,
            y_position=0.0,
            x_offset=0.1,
            y_offset=-0.05,
        ),
        FigurePanel(
            letter="B",
            path=nullcline_reconstruction_paths[dataset_low],
            x_position=2.8,
            y_position=0.0,
            x_offset=0.55,
            y_offset=0.15,
        ),
        FigurePanel(
            letter="C",
            path=contour_plot_paths[dataset_high],
            x_position=0.0,
            y_position=1.5,
            x_offset=0.1,
            y_offset=-0.05,
        ),
        FigurePanel(
            letter="D",
            path=nullcline_reconstruction_paths[dataset_high],
            x_position=2.8,
            y_position=1.5,
            x_offset=0.55,
            y_offset=0.15,
        ),
    ]

    build_figure_from_panels(
        panels, output_path / "supp_fig_nullcline_walks.svg", width=MAX_FIGURE_WIDTH, height=3.0
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
