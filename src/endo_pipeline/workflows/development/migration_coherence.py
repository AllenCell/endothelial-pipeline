from endo_pipeline.cli import Datasets
from endo_pipeline.settings.migration_coherence import DEFAULT_MIGRATION_COHERENCE_FEATURE


def main(
    datasets: Datasets | None = None,
    optical_flow_feature: str = DEFAULT_MIGRATION_COHERENCE_FEATURE,
    plot_fixed_points: bool = False,
) -> None:
    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        check_required_columns_in_dataframe,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.analyze.migration_pc.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.diffae_features.pplane import (
        make_legend_handles_for_fixed_pts,
    )
    from endo_pipeline.library.visualize.migration_coherence import (
        plot_optical_flow_feature_distribution,
        plot_scatter_and_binned_heatmap,
    )
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        STABILITY_COLOR_DICT,
        STABILITY_COLUMN_NAME,
        STABILITY_MARKER_DICT,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # Load diffae features
    crop_pattern = MIGRATION_COHERENCE_CROP_PATTERN
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    feature_dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern=crop_pattern
    )
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # get fit PCA object to apply PCA transformation to diffae features before
    # plotting against optical flow features.
    pca = fit_pca(num_pcs=3)

    fixed_points_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{feature_dataframe_manifest_name}"
    )
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    # Default list of datasets if not provided, only include datasets available
    # in the provided dataframe manifest
    valid_dataset_options = list_datasets_with_dataframes(feature_dataframe_manifest)
    if datasets is None:
        # these collections are mutually exclusive, so we don't have to worry
        # about duplicates when concatenating
        dataset_names = get_datasets_in_collection(
            "diffae_model_training", valid_dataset_options
        ) + get_datasets_in_collection("replicate_2_datasets", valid_dataset_options)
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    # if in demo mode, only process the first dataset and log a warning
    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning(
            "Running in demo mode, only processing first dataset [ %s ]",
            dataset_names[0],
        )

    # Load optical flow features and plot against diffae features
    for dataset_name in dataset_names:
        output_dir = get_output_path(__file__, dataset_name)

        df_dataset = get_dataframe_for_dynamics_workflows(
            dataset_name,
            feature_dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
        )
        df_of = add_optical_flow_features(
            df_dataset,
            datasets=[dataset_name],
        )

        # initialize fixed_points_dataframe to None in case we aren't plotting
        # fixed points or if loading the fixed points dataframe fails for any
        # reason, then try to load the fixed points dataframe if we're plotting fixed points
        fixed_points_dataframe: pd.DataFrame | None = None
        if plot_fixed_points:
            try:
                fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                    fixed_points_dataframe_manifest, dataset_name
                )
                fixed_points_dataframe = load_dataframe(
                    fixed_points_dataframe_location, delay=False
                )
                check_required_columns_in_dataframe(
                    fixed_points_dataframe,
                    required_columns=[
                        *DYNAMICS_COLUMN_NAMES,
                        ColumnName.DATASET,
                        STABILITY_COLUMN_NAME,
                    ],
                )
            except KeyError:
                # if the fixed points dataframe for this dataset isn't found in
                # the manifest, log a warning and continue without loading fixed
                # points (i.e., fixed_points_dataframe will remain None)
                logger.warning(
                    "No fixed point dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. "
                    "Fixed points will not be overlaid on the migration coherence plots for this dataset.",
                    dataset_name,
                    fixed_points_dataframe_manifest.name,
                )

        plot_optical_flow_feature_distribution(
            df=df_of,
            optical_flow_feature=optical_flow_feature,
            datasets=[dataset_name],
            output_dir=output_dir,
            binwidth=0.02,
            bins=50,
            kde=True,
        )
        for x_col, y_col in [
            (ColumnName.POLAR_RADIUS, ColumnName.POLAR_ANGLE),
            (ColumnName.PC3_FLIPPED, ColumnName.POLAR_ANGLE),
            (ColumnName.POLAR_RADIUS, ColumnName.PC3_FLIPPED),
        ]:
            figure_filename = f"{dataset_name}_{x_col}_vs_{y_col}_colored_by_{optical_flow_feature}"
            logger.info(
                "Plotting optical flow feature over [ %s ] vs [ %s ] for dataset [ %s ]",
                x_col,
                y_col,
                dataset_name,
            )
            fig, axs = plot_scatter_and_binned_heatmap(
                df=df_of,
                dataset_name=dataset_name,
                x_col=x_col,
                y_col=y_col,
                color_col=optical_flow_feature,
                vmax=1,
                vmin=0,
                x_bin_size=0.25,
                y_bin_size=0.25,
            )
            plt.show()
            save_plot_to_path(
                fig,
                output_dir,
                figure_filename,
            )
            plt.close(fig)

            # if fixed points are available, overlay them on the scatter plot
            if fixed_points_dataframe is not None:
                for _, row in fixed_points_dataframe.iterrows():
                    stability = row[STABILITY_COLUMN_NAME]
                    marker = STABILITY_MARKER_DICT.get(stability, "o")
                    color = STABILITY_COLOR_DICT.get(stability, "gray")
                    axs[1].scatter(
                        row[x_col],
                        row[y_col],
                        marker=marker,
                        color=color,
                        edgecolor="black",
                        s=100,
                        label=f"Fixed Point ({stability})",
                    )
                # add legend for fixed points
                legend_handles = make_legend_handles_for_fixed_pts(
                    fixed_points_dataframe[STABILITY_COLUMN_NAME].unique().tolist(),
                    face_color_dict=STABILITY_COLOR_DICT,
                    marker_dict=STABILITY_MARKER_DICT,
                    marker_size=10,
                    edge_color="black",
                )
                fig.legend(
                    handles=legend_handles,
                    bbox_to_anchor=(1.05, 0.95),
                    title="fixed point stability",
                    loc="upper left",
                    fontsize=10,
                )
                fig.tight_layout()
                save_plot_to_path(
                    fig,
                    output_dir,
                    f"{figure_filename}_with_fixed_points",
                )
                plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
