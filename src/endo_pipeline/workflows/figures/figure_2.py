from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Compile panels for Figure 2.

    - **Panel A**: 3D visualizations of drift vector field and nullclines for
      example low shear stress dataset.
    - **Panel B**: 1D plot of drift along polar angle coordinate for example low
      shear stress dataset, 2D contour plot of drift coefficients for polar
      radius and rho (-PC3) coordinates for example low shear stress dataset.
    - **Panel C**: DiffAE generated synthetic images along nullcline in polar
      radius and rho (-PC3) coordinates for example low shear stress dataset.
    - **Panel D-F**: Same as A-C for example high shear stress dataset.
    - **Panel G**: Summary plot of fixed point locations across multiple
      datasets, colored by migration coherence (EMA-smoothed optical flow unit
      vector mean).
    - **Panel H**: Schematic of first passage time calculation for example
      trajectories in low shear stress dataset.
    - **Panel I**: Histogram of first passage time correlation coefficients
      across multiple datasets.

    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, load_model
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
        load_drift_dataframe_for_dataset,
    )
    from endo_pipeline.library.visualize.figure_2 import (
        make_1d_drift_plot_panel,
        make_2d_contour_plot_panel,
        make_3d_vector_field_plot_panel,
        make_first_passage_time_distance_to_linefit_hist,
        reconstruct_fixed_points,
    )
    from endo_pipeline.library.visualize.figure_fpt import generate_first_passage_time_example
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_fixed_point_dataset_summary,
        plot_cross_dataset_summaries,
    )
    from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameSuffix
    from endo_pipeline.settings.examples import EXAMPLE_DATASET, FPT_FIG_EXAMPLES
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    )
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D", "E", "F"])

    # figure is for grid based crops
    crop_pattern = "grid"

    dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
    dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]
    dataset_summary_list = SUMMARY_PLOT_DATASETS["low_high"]

    columns_r_rho = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    columns_r_rho_str = join_sorted_strings(columns_r_rho)
    columns_r_rho_fixed_point = [f"{col}{ColumnNameSuffix.FIXED_POINTS}" for col in columns_r_rho]
    column_theta = Column.DiffAEData.POLAR_ANGLE
    column_theta_fixed_point = f"{column_theta}{ColumnNameSuffix.FIXED_POINTS}"
    optical_flow_feature = Column.OpticalFlow.UNIT_VECTOR_MEAN
    feature_column_names = [column_theta, *columns_r_rho]
    feature_columns_str = join_sorted_strings(feature_column_names)

    # load dataframe manifests for diffae features, fixed points, optical flow
    # features, and bootstrapped fixed points for this crop pattern, which will be
    # used for all visualizations in this figure
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
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
    bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[crop_pattern]
    bootstrap_dataframe_manifest = load_dataframe_manifest(bootstrap_dataframe_manifest_name)

    # get labels for provided set of feature columns
    columns_for_summary_plots = [*feature_column_names, optical_flow_feature]
    column_labels_r_rho = [COLUMN_METADATA[column].label for column in columns_r_rho]
    column_label_theta = COLUMN_METADATA[column_theta].label

    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
    theta_plot_paths: dict[str, Path] = {}
    contour_plot_paths: dict[str, Path] = {}
    fixed_point_reconstruction_paths: dict[str, Path] = {}
    vector_field_plot_paths: dict[str, Path] = {}
    for dataset_name, arrow_scale_1d, arrow_width_1d, include_colorbar, include_legend in [
        (dataset_low, 1.5, 0.05, True, False),
        (dataset_high, 0.5, 0.05, False, True),
    ]:
        fig_savedir = get_output_path(__file__, dataset_name)
        dataset_config = load_dataset_config(dataset_name)
        shear_stress_bin = dataset_config.flow_conditions[-1].shear_stress_bin
        shear_stress_label = f"{shear_stress_bin} dyn/cm{Unicode.SQUARED}"

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

        drift_theta_dataframe = load_drift_dataframe_for_dataset(
            dataset_name, columns=[column_theta]
        )
        drift_theta, centers_theta = get_reshaped_vector_field_and_grid(
            drift_theta_dataframe, column_names=[column_theta]
        )
        stable_fixed_point_theta = stable_fixed_points_dict[column_theta][
            column_theta_fixed_point
        ].to_numpy()

        vector_field_plot_paths[dataset_name] = make_3d_vector_field_plot_panel(
            figure_size=(1.55, 2.25),
            output_path=fig_savedir,
            dataset_name=dataset_name,
            shear_stress_label=shear_stress_label,
            include_legend=include_legend,
            include_colorbar=include_colorbar,
            **placeholders["A"],
        )

        # plot 1D drift in theta and save
        theta_plot_paths[dataset_name] = make_1d_drift_plot_panel(
            figure_size=(1.25, 1.25),
            output_path=fig_savedir,
            shear_stress_label=shear_stress_label,
            drift=drift_theta,
            theta_values=centers_theta[-1],
            column_label=column_label_theta,
            stable_fixed_point=stable_fixed_point_theta,
            filename=f"{dataset_name}_{Column.DiffAEData.POLAR_ANGLE}_drift",
            arrow_scale=arrow_scale_1d,
            arrow_width=arrow_width_1d,
            include_legend=include_legend,
            **placeholders["B"],
        )

        contour_plot_paths[dataset_name], _ = make_2d_contour_plot_panel(
            figure_size=(1.7, 2.83),
            output_path=fig_savedir,
            drift=drift_r_rho,
            meshgrid=centers_mesh,
            column_labels=column_labels_r_rho,
            stable_fixed_point=stable_fixed_point_r_rho,
            filename=f"{dataset_name}_{columns_r_rho_str}_contours",
            include_legend=include_legend,
            include_colorbar=include_colorbar,
            **placeholders["B"],
        )

        fixed_point_reconstruction_paths[dataset_name] = reconstruct_fixed_points(
            fixed_point_df=stable_fixed_points_dict[feature_columns_str],
            shear_stress_label=shear_stress_label,
            model=model,
            output_path=fig_savedir,
            figure_size=(1.0, 1.2),
            num_gpus=NUM_GPUS,
            include_row_label=include_colorbar,
            **placeholders["C"],
        )

    # --- Cross-dataset summary plots ---
    fixed_point_summary_df = build_dataframe_for_fixed_point_dataset_summary(
        dataset_names=dataset_summary_list,
        feature_dataframe_manifest=feature_dataframe_manifest,
        bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
        column_names=columns_for_summary_plots,
        convert_angle_to_nematic=False,
        unwrap_angle=True,
        stable_only=True,
    )
    # summary plot of fixed point locations across datasets
    fixed_point_summary_plot_path = plot_cross_dataset_summaries(
        fixed_point_summary_df,
        output_path=output_path,
        column_names=feature_column_names,
        axis_mode="shear_stress",
        subplot_layout="horizontal",
        figure_size=(3.15, 1.8),
        color_by_column=Column.OpticalFlow.UNIT_VECTOR_MEAN,
        ylabel_rotation=0,
        remove_label_linebreaks=False,
        **placeholders["D"],
    )
    # --- First passage time analysis schematic ---
    low_flow_dataset = FPT_FIG_EXAMPLES["low_flow"]
    trajectory_example_filepath = generate_first_passage_time_example(
        dataset_name=low_flow_dataset.dataset_name,
        example_fixed_point_index=low_flow_dataset.fixed_point_index,
        example_tracked_crop_index=low_flow_dataset.tracked_crop_index,
        example_grid_crop_index=low_flow_dataset.grid_crop_index,
        output_path=output_path,
        figure_size=(1.85, 1.95),
        **placeholders["E"],
    )
    # --- Histogram of first passage time correlation ---
    first_passage_path = make_first_passage_time_distance_to_linefit_hist(
        figure_size=(2.5, 1.25),
        output_path=output_path,
        dataset_names=dataset_summary_list,
        weighted=False,
        **placeholders["F"],
    )

    # --- Assemble all panels into final figure ---
    panels = [
        # --- 3D plots (panel A) ---
        FigurePanel(
            letter="A",
            path=vector_field_plot_paths[dataset_low],
            x_position=0.0,
            y_position=0.0,
            x_offset=0.05,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=vector_field_plot_paths[dataset_high],
            x_position=1.7,
            y_position=0.0,
            x_offset=0.0,
            y_offset=0.0,
        ),
        # --- 1D and contour plots (panel B) ---
        FigurePanel(
            letter="B",
            path=theta_plot_paths[dataset_low],
            x_position=0.0,
            y_position=2.05,
            x_offset=0.26,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=contour_plot_paths[dataset_low],
            x_position=0.0,
            y_position=3.6,
            x_offset=0.0,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=theta_plot_paths[dataset_high],
            x_position=1.91,
            y_position=2.05,
            x_offset=0.00,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=contour_plot_paths[dataset_high],
            x_position=1.65,
            y_position=3.6,
            x_offset=0.0,
            y_offset=0.0,
        ),
        # --- Fixed point reconstructions (panel C) ---
        FigurePanel(
            letter="C",
            path=fixed_point_reconstruction_paths[dataset_low],
            x_position=3.225,
            y_position=0.0,
            x_offset=0.3,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=fixed_point_reconstruction_paths[dataset_high],
            x_position=4.825,
            y_position=0.0,
            x_offset=0.3,
            y_offset=0.0,
        ),
        # --- Remaining rows: summary plots, first passage time results ---
        FigurePanel(
            letter="D",
            path=fixed_point_summary_plot_path,
            x_position=3.25,
            y_position=1.35,
            x_offset=0.05,
            y_offset=0.15,
        ),
        FigurePanel(
            letter="E",
            path=trajectory_example_filepath,
            x_position=3.25,
            y_position=3.35,
            x_offset=0.3,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="F",
            path=first_passage_path,
            x_position=3.25,
            y_position=5.2,
            x_offset=0.1,
            y_offset=0.05,
        ),
    ]

    build_figure_from_panels(
        panels, output_path / "figure_2.svg", width=MAX_FIGURE_WIDTH, height=6.5
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
