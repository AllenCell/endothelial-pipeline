def main() -> None:
    """
    Compile panels for supplementary figure comparing VE-cadherin
    reconstructions at given stable fixed points to real examples.

    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, load_model
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.supp_fig_real_examples import (
        make_real_and_reconstructed_example_contact_sheet,
    )
    from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        StabilityLabel,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    plt.style.use("endo_pipeline.figure")

    base_output_dir = get_output_path(__file__)

    # figure is for grid based crops
    crop_pattern = "grid"

    dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
    dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]

    columns_r_rho = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    columns_r_rho_str = join_sorted_strings(columns_r_rho)
    column_theta = Column.DiffAEData.POLAR_ANGLE
    feature_column_names = [column_theta, *columns_r_rho]
    feature_columns_str = join_sorted_strings(feature_column_names)

    # load dataframe manifests for diffae features, fixed points, optical flow
    # features, and bootstrapped fixed points for this crop pattern, which will be
    # used for all visualizations in this figure
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]
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
    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
    crop_contact_sheet_paths: dict[str, Path] = {}
    for dataset_name, include_colorbar, include_legend, arrow_scale_1d, contact_sheet_title in [
        (dataset_low, True, True, 3.25, "Reconstructed \nVE-cadherin at\nstable fixed point"),
        (dataset_high, False, False, 1.0, None),
    ]:
        fig_savedir = get_output_path(__file__, dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        feature_dataframe = load_dataframe(feature_dataframe_manifest.locations[dataset_name])

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

        # make contact sheet of example ve-cadherin reconstruction at stable
        # fixed points for this dataset
        crop_contact_sheet_paths[dataset_name] = make_real_and_reconstructed_example_contact_sheet(
            dataset_config=dataset_config,
            stable_fixed_point_dataframe=stable_fixed_points_dict[feature_columns_str],
            crop_features_dataframe=feature_dataframe,
            feature_column_names=feature_column_names,
            model=model,
            n_crop_examples=3,
            fig_savedir=fig_savedir,
            fig_filename=f"{dataset_name}_crop_examples",
            file_format=".svg",
            gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
            fig_kwargs={"figsize": (MAX_FIGURE_WIDTH / 2 - 0.2, 3), "layout": "constrained"},
            random_seed=7,
            num_gpus=NUM_GPUS,
        )

    # --- Assemble all panels into final figure ---
    panels = [
        # --- Low flow dataset ---
        FigurePanel(
            letter="A",
            path=crop_contact_sheet_paths[dataset_low],
            x_position=0.0,
            y_position=0.0,
            x_offset=0.08,
            y_offset=0.12,
        ),
        # --- High flow dataset  ---
        FigurePanel(
            letter="B",
            path=crop_contact_sheet_paths[dataset_high],
            x_position=MAX_FIGURE_WIDTH / 2,
            y_position=0.0,
            x_offset=0.08,
            y_offset=0.12,
        ),
    ]

    build_figure_from_panels(
        panels, base_output_dir / "supp_fig_real_examples.svg", width=MAX_FIGURE_WIDTH, height=3.25
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
