def main() -> None:
    """Compile panels for Figure 2."""
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
        make_2d_quiver_plot_panel,
        make_crop_example_contact_sheet,
        reconstruct_along_nullcline,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_first_passage_time_dataset_summary,
        build_dataframe_for_fixed_point_dataset_summary,
        plot_cross_dataset_summaries,
    )
    from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import POLAR_ANGLE_RANGE
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.figures import FONTSIZE_XSMALL, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.first_passage_time import (
        FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME,
    )
    from endo_pipeline.settings.flow_field_2d import DRIFT_CONTOUR_VMAX, DRIFT_CONTOUR_VMIN
    from endo_pipeline.settings.flow_field_dataframes import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        StabilityLabel,
    )
    from endo_pipeline.settings.flow_field_figure import (
        AXES_LIMITS_2D,
        GRIDSPEC_KWARGS,
        NULLCLINE_STYLES_2D,
        XLABEL_KWARGS,
        YLABEL_KWARGS,
    )
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    plt.style.use("endo_pipeline.figure")

    base_output_dir = get_output_path("figure_2")

    # figure is for grid based crops
    crop_pattern = "grid"

    dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
    dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]
    dataset_summary_list = SUMMARY_PLOT_DATASETS["low_high"]

    columns_r_rho = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    columns_r_rho_str = join_sorted_strings(columns_r_rho)
    column_theta = Column.DiffAEData.POLAR_ANGLE
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
    nullcline_reconstruction_paths: dict[str, list[Path]] = {}
    quiver_plot_paths: dict[str, Path] = {}
    crop_contact_sheet_paths: dict[str, Path] = {}
    for dataset_name, include_legend, arrow_scale_1d in [
        (dataset_low, True, 1.5),
        (dataset_high, False, 1.25),
    ]:
        fig_savedir = get_output_path("figure_2", dataset_name)
        dataset_config = load_dataset_config(dataset_name)
        shear_stress = dataset_config.flow_conditions[-1].shear_stress_bin
        shear_stress_label = f"{shear_stress} dyn/cm{Unicode.SQUARED}"

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
            columns_r_rho
        ].to_numpy()

        drift_theta_dataframe = load_drift_dataframe_for_dataset(
            dataset_name, columns=[column_theta]
        )
        drift_theta, centers_theta = get_reshaped_vector_field_and_grid(
            drift_theta_dataframe, column_names=[column_theta]
        )
        stable_fixed_point_theta = stable_fixed_points_dict[column_theta][column_theta].to_numpy()

        # plot 1D drift in theta and save
        theta_plot_paths[dataset_name] = make_1d_drift_plot_panel(
            drift=drift_theta,
            theta_values=centers_theta[-1],
            column_label=column_label_theta,
            stable_fixed_point=stable_fixed_point_theta,
            figsize=(1.25, 1.5),
            fig_savedir=fig_savedir,
            filename=f"{dataset_name}_{Column.DiffAEData.POLAR_ANGLE}_drift",
            shear_stress_label=shear_stress_label,
            axes_xlim=POLAR_ANGLE_RANGE,
            axes_ylim=(-0.4, 0.4),
            axes_xticks=[0, np.pi / 2, np.pi],
            axes_xtick_labels=[
                f"0={Unicode.PI}",
                f"{Unicode.PI}/2",
                f"{Unicode.PI}=0",
            ],
            axes_yticks=[-0.3, 0.0, 0.3],
            arrow_scale=arrow_scale_1d,
            drift_line_kwargs={"color": "k", "linewidth": 2},
            zero_line_kwargs={"linestyle": "--", "color": "gray", "linewidth": 1, "alpha": 0.7},
            gridspec_kwargs=GRIDSPEC_KWARGS,
            xlabel_kwargs=XLABEL_KWARGS,
            ylabel_kwargs=YLABEL_KWARGS,
        )

        contour_plot_paths[dataset_name], nullcline_coordinates = make_2d_contour_plot_panel(
            drift=drift_r_rho,
            meshgrid=centers_mesh,
            column_labels=column_labels_r_rho,
            figsize=(1.9, 1.25),
            fig_savedir=fig_savedir,
            filename=f"{dataset_name}_{columns_r_rho_str}_contours",
            r_lims=AXES_LIMITS_2D[Column.DiffAEData.POLAR_RADIUS],
            rho_lims=AXES_LIMITS_2D[Column.DiffAEData.PC3_FLIPPED],
            r_ticks=[0.4, 1.0, 1.6],
            rho_ticks=[-0.75, 0.0, 0.75],
            nullcline_r_style=NULLCLINE_STYLES_2D[Column.DiffAEData.POLAR_RADIUS],
            nullcline_rho_style=NULLCLINE_STYLES_2D[Column.DiffAEData.PC3_FLIPPED],
            nullcline_opacity=1.0,
            gridspec_kwargs=GRIDSPEC_KWARGS,
            xlabel_kwargs=XLABEL_KWARGS,
            ylabel_kwargs=YLABEL_KWARGS,
            axes_title_kwargs={
                "fontsize": FONTSIZE_XSMALL,
                "x": 0.05,
                "y": 0.75,
                "rotation": 0,
                "ha": "left",
                "va": "center",
                "bbox": {
                    "boxstyle": "round",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            },
            include_colorbar=True,
        )

        nullcline_reconstruction_paths[dataset_name] = reconstruct_along_nullcline(
            nullcline_coords=nullcline_coordinates,
            theta_value=stable_fixed_point_theta[0],
            model=model,
            fig_savedir=fig_savedir,
            num_gpus=NUM_GPUS,
        )

        quiver_plot_paths[dataset_name] = make_2d_quiver_plot_panel(
            drift=drift_r_rho,
            meshgrid=centers_mesh,
            column_labels=column_labels_r_rho,
            stable_fixed_point=stable_fixed_point_r_rho,
            figsize=(2.05, 1.65),
            fig_savedir=fig_savedir,
            filename=f"{dataset_name}_{columns_r_rho_str}_quiver",
            r_lims=AXES_LIMITS_2D[Column.DiffAEData.POLAR_RADIUS],
            rho_lims=AXES_LIMITS_2D[Column.DiffAEData.PC3_FLIPPED],
            r_ticks=[0.25, 0.75, 1.25, 1.75],
            rho_ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
            nullcline_r_style=NULLCLINE_STYLES_2D[Column.DiffAEData.POLAR_RADIUS],
            nullcline_rho_style=NULLCLINE_STYLES_2D[Column.DiffAEData.PC3_FLIPPED],
            nullcline_opacity=0.9,
            quiver_color="dimgrey",
            quiver_scale=3.5,
            quiver_downsample=4,
            vmin=DRIFT_CONTOUR_VMIN,
            vmax=DRIFT_CONTOUR_VMAX,
            include_legend=include_legend,
            gridspec_kwargs=GRIDSPEC_KWARGS,
            xlabel_kwargs=XLABEL_KWARGS,
            ylabel_kwargs=YLABEL_KWARGS,
            quiver_legend_kwargs={
                "fontsize": "xx-small",
                "title_fontsize": "xx-small",
                "loc": "upper center",
                "bbox_to_anchor": (0.5, 1.25),
                "ncol": 2,
                "handletextpad": 0.3,
            },
        )

        # make contact sheet of example ve-cadherin reconstruction at stable
        # fixed points for this dataset
        crop_contact_sheet_paths[dataset_name] = make_crop_example_contact_sheet(
            stable_fixed_point_dataframe=stable_fixed_points_dict[feature_columns_str],
            feature_column_names=feature_column_names,
            model=model,
            n_crop_examples=2,
            fig_savedir=fig_savedir,
            fig_filename=f"{dataset_name}_crop_examples",
            file_format=".svg",
            gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
            fig_kwargs={"figsize": (0.85, 1.45), "layout": "constrained"},
            random_seed=7,
            num_gpus=NUM_GPUS,
        )

    # --- Cross-dataset summary plots ---
    fixed_point_summary_df = build_dataframe_for_fixed_point_dataset_summary(
        dataset_names=dataset_summary_list,
        feature_dataframe_manifest=feature_dataframe_manifest,
        bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
        column_names=columns_for_summary_plots,
        convert_angle_to_nematic=True,
        stable_only=True,
    )
    # panel E: summary plot of fixed point locations across datasets
    fixed_point_summary_plot_path = plot_cross_dataset_summaries(
        fixed_point_summary_df,
        output_dir=base_output_dir,
        column_names=feature_column_names,
        axis_mode="shear_stress",
        figure_size=(3.5, 2),
    )
    # panel F: summary plot of migration coherence at fixed points across
    # datasets
    migration_summary_plot_path = plot_cross_dataset_summaries(
        fixed_point_summary_df,
        output_dir=base_output_dir,
        column_names=[optical_flow_feature],
        axis_mode="shear_stress",
        figure_size=(1.25, 2),
    )

    # --- Histogram of first passage time correlation ---
    fpt_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME)
    first_passage_summary_df = build_dataframe_for_first_passage_time_dataset_summary(
        dataset_names=dataset_summary_list, first_passage_time_manifest=fpt_manifest
    )
    first_passage_path = plot_cross_dataset_summaries(
        first_passage_summary_df,
        output_dir=base_output_dir,
        column_names=[Column.VectorField.PEARSON_R],
        axis_mode="shear_stress",
        figure_size=(1.125, 2.05),
        set_y_lims=True,
    )

    # --- Assemble all panels into final figure ---
    panels = [
        # --- Low flow dataset (row 1) ---
        FigurePanel(
            letter="A",
            path=theta_plot_paths[dataset_low],
            x_position=0,
            y_position=0.00,
            x_offset=0.05,
            y_offset=-0.05,
        ),
        FigurePanel(
            letter="",
            path=contour_plot_paths[dataset_low],
            x_position=1.6,
            y_position=0.00,
            x_offset=0.0,
            y_offset=-0.05,
        ),
        # FigurePanel(  # r nullcline for low flow dataset
        #     letter="",
        #     path=nullcline_reconstruction_paths[dataset_low][0],
        #     x_position=3.0,
        #     y_position=0.1,
        #     x_offset=0.1,
        #     y_offset=0.0,
        # ),
        # FigurePanel(  # rho nullcline for low flow dataset
        #     letter="",
        #     path=nullcline_reconstruction_paths[dataset_low][1],
        #     x_position=3.55,
        #     y_position=0.1,
        #     x_offset=0.1,
        #     y_offset=0.0,
        # ),
        FigurePanel(
            letter="",
            path=quiver_plot_paths[dataset_low],
            x_position=3.9,
            y_position=0.05,
            x_offset=-0.1,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="B",
            path=crop_contact_sheet_paths[dataset_low],
            x_position=MAX_FIGURE_WIDTH - 1.0,
            y_position=0.05,
            x_offset=0.15,
            y_offset=0.0,
        ),
        # --- High flow dataset (row 2) ---
        FigurePanel(
            letter="C",
            path=theta_plot_paths[dataset_high],
            x_position=0,
            y_position=1.9,
            x_offset=0.5,
            y_offset=-0.05,
        ),
        FigurePanel(
            letter="",
            path=contour_plot_paths[dataset_high],
            x_position=1.6,
            y_position=1.9,
            x_offset=0.0,
            y_offset=-0.05,
        ),
        # FigurePanel(  # r nullcline for high flow dataset
        #     letter="",
        #     path=nullcline_reconstruction_paths[dataset_high][0],
        #     x_position=3.0,
        #     y_position=2.15,
        #     x_offset=0.1,
        #     y_offset=0.0,
        # ),
        # FigurePanel(  # rho nullcline for high flow dataset
        #     letter="",
        #     path=nullcline_reconstruction_paths[dataset_high][1],
        #     x_position=3.5,
        #     y_position=2.15,
        #     x_offset=0.1,
        #     y_offset=0.00,
        # ),
        FigurePanel(
            letter="",
            path=quiver_plot_paths[dataset_high],
            x_position=3.9,
            y_position=2.05,
            x_offset=-0.1,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="D",
            path=crop_contact_sheet_paths[dataset_high],
            x_position=MAX_FIGURE_WIDTH - 1.0,
            y_position=2.05,
            x_offset=0.15,
            y_offset=0.05,
        ),
        # --- Bottom row: first passage time and summary plots ---
        FigurePanel(
            letter="E",
            path=fixed_point_summary_plot_path,
            x_position=0.0,
            y_position=4.0,
            x_offset=0,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="F",
            path=migration_summary_plot_path,
            x_position=3.6,
            y_position=4.0,
            x_offset=0.1,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="G",
            path=first_passage_path,
            x_position=5.175,
            y_position=4.0,
            x_offset=0.05,
            y_offset=0.15,
        ),
    ]

    build_figure_from_panels(
        panels, base_output_dir / "figure_2.svg", width=MAX_FIGURE_WIDTH, height=6.2
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
