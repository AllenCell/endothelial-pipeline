def main() -> None:
    """Compile panels for Figure 2."""
    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import (
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        load_model,
        save_plot_to_path,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_stability,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
        load_drift_dataframe_for_dataset,
    )
    from endo_pipeline.library.visualize.diffae_features.dynamics import plot_contour_colorbar
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
    from endo_pipeline.library.visualize.figure_2 import (
        make_1d_drift_plot_panel,
        make_2d_contour_plot_panel,
        make_2d_quiver_plot_panel,
        make_crop_example_contact_sheet,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.migration_coherence import plot_optical_flow_histogram
    from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
    from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        METADATA_COLUMNS_TO_KEEP,
        POLAR_ANGLE_RANGE,
    )
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_2d import (
        DRIFT_CONTOUR_CBAR_NUM_TICKS,
        DRIFT_CONTOUR_CBAR_ROUND,
        DRIFT_CONTOUR_COLORMAP,
        DRIFT_CONTOUR_VMAX,
        DRIFT_CONTOUR_VMIN,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING,
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
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
    fixed_points_r_rho_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{columns_r_rho_str}_{base_name}"
    )
    fixed_points_r_rho_dataframe_manifest = load_dataframe_manifest(
        fixed_points_r_rho_dataframe_manifest_name
    )
    fixed_points_theta_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{column_theta}_{base_name}"
    )
    fixed_points_theta_dataframe_manifest = load_dataframe_manifest(
        fixed_points_theta_dataframe_manifest_name
    )
    fixed_points_3d_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{feature_columns_str}_{base_name}"
    )
    fixed_points_3d_dataframe_manifest = load_dataframe_manifest(
        fixed_points_3d_dataframe_manifest_name
    )
    bootstrap_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"
    bootstrap_dataframe_manifest = load_dataframe_manifest(bootstrap_dataframe_manifest_name)

    # get labels for provided set of feature columns
    columns_for_summary_plots = [*feature_column_names, optical_flow_feature]
    column_labels_r_rho = [COLUMN_METADATA[column].label for column in columns_r_rho]
    column_label_theta = COLUMN_METADATA[column_theta].label
    dataframe_columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *feature_column_names]

    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    # make svg of just the colorbar with set ticks and extended on both sides
    fig, ax = plot_contour_colorbar(
        figsize=(0.75, MAX_FIGURE_WIDTH / 4),
        vmin=DRIFT_CONTOUR_VMIN,
        vmax=DRIFT_CONTOUR_VMAX,
        num_ticks=DRIFT_CONTOUR_CBAR_NUM_TICKS,
        tick_label_round=DRIFT_CONTOUR_CBAR_ROUND,
        extend="both",
        colormap=DRIFT_CONTOUR_COLORMAP,
        orientation="vertical",
    )
    save_plot_to_path(
        fig, base_output_dir, "colorbar", file_format=".svg", transparent=True, tight_layout=True
    )

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
    contour_plot_paths: dict[str, Path] = {}
    quiver_plot_paths: dict[str, Path] = {}
    theta_plot_paths: dict[str, Path] = {}
    crop_contact_sheet_paths: dict[str, Path] = {}
    for dataset_name, include_legend, arrow_scale_1d in [
        (dataset_low, True, 3.25),
        (dataset_high, False, 1.0),
    ]:
        fig_savedir = get_output_path("figure_2", dataset_name)
        dataset_config = load_dataset_config(dataset_name)
        shear_stress = math.ceil(max(fc.shear_stress for fc in dataset_config.flow_conditions))
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

        contour_plot_paths[dataset_name] = make_2d_contour_plot_panel(
            drift=drift_r_rho,
            meshgrid=centers_mesh,
            column_labels=column_labels_r_rho,
            figsize=(1.75, 1.9),
            fig_savedir=fig_savedir,
            filename=f"{dataset_name}_{columns_r_rho_str}_contours",
            shear_stress_label=shear_stress_label,
            r_lims=AXES_LIMITS_2D[Column.DiffAEData.POLAR_RADIUS],
            rho_lims=AXES_LIMITS_2D[Column.DiffAEData.PC3_FLIPPED],
            r_ticks=[0.25, 1.0, 1.75],
            rho_ticks=[-0.75, 0.0, 0.75],
            nullcline_r_style=NULLCLINE_STYLES_2D[Column.DiffAEData.POLAR_RADIUS],
            nullcline_rho_style=NULLCLINE_STYLES_2D[Column.DiffAEData.PC3_FLIPPED],
            nullcline_opacity=1.0,
            gridspec_kwargs=GRIDSPEC_KWARGS,
            xlabel_kwargs=XLABEL_KWARGS,
            ylabel_kwargs=YLABEL_KWARGS,
            axes_title_kwargs={
                "fontsize": "small",
                "x": 1.05,
                "y": 0.5,
                "rotation": 0,
                "ha": "left",
                "va": "center",
            },
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

        # plot 1D drift in theta and save
        theta_plot_paths[dataset_name] = make_1d_drift_plot_panel(
            drift=drift_theta,
            theta_values=centers_theta[-1],
            column_label=column_label_theta,
            stable_fixed_point=stable_fixed_point_theta,
            figsize=(MAX_FIGURE_WIDTH / 4, MAX_FIGURE_HEIGHT / 4),
            fig_savedir=fig_savedir,
            filename=f"{dataset_name}_{Column.DiffAEData.POLAR_ANGLE}_drift",
            axes_xlim=POLAR_ANGLE_RANGE,
            axes_ylim=(-0.4, 0.4),
            axes_xticks=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
            axes_xtick_labels=[
                f"0={Unicode.PI}",
                f"{Unicode.PI}/4",
                f"{Unicode.PI}/2",
                f"3{Unicode.PI}/4",
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

        # make contact sheet of example crops at stable fixed points for this
        # dataset (panel below the flow field visualizations)
        feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])
        dataframe_steady_state = filter_dataframe_to_steady_state(feature_dataframe, dataset_config)
        crop_contact_sheet_paths[dataset_name] = make_crop_example_contact_sheet(
            dataset_config=dataset_config,
            stable_fixed_point_dataframe=stable_fixed_points_dict[feature_columns_str],
            crop_features_dataframe=dataframe_steady_state,
            feature_column_names=feature_column_names,
            model=model,
            n_crop_examples=2,
            fig_savedir=fig_savedir,
            fig_filename=f"{dataset_name}_crop_examples",
            file_format=".svg",
            gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
            fig_kwargs={"figsize": (MAX_FIGURE_WIDTH / 2 - 0.2, 2), "layout": "constrained"},
            random_seed=7,
            num_gpus=NUM_GPUS,
        )

    # --- Cross-dataset summary plots ---
    plot_cross_dataset_summaries(
        dataset_names=dataset_summary_list,
        feature_dataframe_manifest=feature_dataframe_manifest,
        fixed_points_bootstrap_dataframe_manifest=bootstrap_dataframe_manifest,
        output_dir=base_output_dir,
        column_names=columns_for_summary_plots,
        x_axis_mode="shear_stress_categorical",
        figure_size=(MAX_FIGURE_WIDTH - 2.1, 2),
        stable_only=True,
    )

    fig, ax = plt.subplots(figsize=(2, 2), layout="constrained")
    for dataset_name in [dataset_low, dataset_high]:
        # get settings
        dataset_config = load_dataset_config(dataset_name)
        shear_stress = math.ceil(max(fc.shear_stress for fc in dataset_config.flow_conditions))

        # load and filter data
        df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df_ = df[dataframe_columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df_, dataset_config)

        df_of = add_optical_flow_features(
            df_steady_state,
            datasets=[dataset_name],
        )

        fig = plot_optical_flow_histogram(
            df=df_of,
            optical_flow_feature=optical_flow_feature,
            feature_label="Migration Coherence",
            feature_lim=(0, 1),
            ss_label=f"{shear_stress} dyn/cm{Unicode.SQUARED}",
            color=get_dataset_color(dataset_name),
            df_fp=None,
            binwidth=0.02,
            figure=(fig, ax),
            legend_loc=None,
        )
    save_plot_to_path(
        fig,
        base_output_dir,
        "migration_coherence_distribution_high_low_flow_comparison",
        pad_inches=0,
        tight_layout=False,
        file_format=".svg",
    )

    # --- Assemble all panels into final figure ---
    panels = [
        # --- Low flow dataset (row 1) ---
        FigurePanel(
            letter="A",
            path=contour_plot_paths[dataset_low],
            x_position=0,
            y_position=0.0,
            x_offset=0.15,
            y_offset=-0.1,
        ),
        FigurePanel(
            letter="",
            path=base_output_dir / "colorbar.svg",
            x_position=MAX_FIGURE_WIDTH / 4 - 0.1,
            y_position=0.0,
            x_offset=0.08,
            y_offset=0.00,
        ),
        FigurePanel(
            letter="",
            path=quiver_plot_paths[dataset_low],
            x_position=MAX_FIGURE_WIDTH / 4 + 0.9,
            y_position=0.0,
            x_offset=-0.1,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="B",
            path=theta_plot_paths[dataset_low],
            x_position=3 * MAX_FIGURE_WIDTH / 4 - 0.35,
            y_position=0.0,
            x_offset=0.4,
            y_offset=-0.2,
        ),
        # --- High flow dataset (row 2) ---
        FigurePanel(
            letter="C",
            path=contour_plot_paths[dataset_high],
            x_position=0,
            y_position=1.85,
            x_offset=0.15,
            y_offset=-0.05,
        ),
        FigurePanel(
            letter="",
            path=base_output_dir / "colorbar.svg",
            x_position=MAX_FIGURE_WIDTH / 4 - 0.1,
            y_position=1.85,
            x_offset=0.08,
            y_offset=0,
        ),
        FigurePanel(
            letter="",
            path=quiver_plot_paths[dataset_high],
            x_position=MAX_FIGURE_WIDTH / 4 + 0.9,
            y_position=1.85,
            x_offset=-0.1,
            y_offset=-0.1,
        ),
        FigurePanel(
            letter="D",
            path=theta_plot_paths[dataset_high],
            x_position=3 * MAX_FIGURE_WIDTH / 4 - 0.35,
            y_position=1.85,
            x_offset=0.4,
            y_offset=-0.2,
        ),
        # --- Contact sheets (row 3) ---
        FigurePanel(
            letter="E",
            path=crop_contact_sheet_paths[dataset_low],
            x_position=0.0,
            y_position=3.8,
            x_offset=0.08,
            y_offset=0.12,
        ),
        FigurePanel(
            letter="F",
            path=crop_contact_sheet_paths[dataset_high],
            x_position=MAX_FIGURE_WIDTH / 2,
            y_position=3.8,
            x_offset=0.08,
            y_offset=0.12,
        ),
        # --- Bottom row ---
        FigurePanel(
            letter="G",
            path=base_output_dir / "migration_coherence_distribution_high_low_flow_comparison.svg",
            x_position=0,
            y_position=6.0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="H",
            path=base_output_dir
            / "nematic_order_polar_r_rho_ema01_optical_flow_mean_unit_vector_dt1_fp_vs_shear_stress.svg",
            x_position=2.1,
            y_position=6.0,
            x_offset=0,
            y_offset=0,
        ),
    ]

    build_figure_from_panels(
        panels, base_output_dir / "figure_2.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
