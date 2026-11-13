from endo_pipeline.cli import Datasets
from endo_pipeline.settings.first_passage_time import (
    FIRST_PASSAGE_TIME_MIN_NUM_TRAJECTORIES_PER_BIN,
)


def main(
    datasets: Datasets | None = None,
    min_num_traj_per_bin: int = FIRST_PASSAGE_TIME_MIN_NUM_TRAJECTORIES_PER_BIN,
) -> None:
    """
    Visualize first passage time results from `compute-first-passage-time`.

    #first-passage-time #grid-based #cell-centered #visualization #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-first-passage-time -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-first-passage-time --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize
    first passage time for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to visualize first passage time.
    min_num_traj_per_bin
        Minimum number of trajectories per bin.
    """

    import logging
    from typing import Any

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.first_passage_time import (
        build_first_passage_time_line_fit_results_dataframe,
        filter_first_passage_time_by_min_num_trajectories,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_first_passage_time_3d_scatter,
        plot_first_passage_time_correlation_summary,
        plot_first_passage_time_correlations,
        plot_first_passage_time_histogram,
        plot_first_passage_time_parameter_sweep,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.first_passage_time import (
        FIRST_PASSAGE_TIME_PARAMETER_SWEEP_MANIFEST_NAME,
        FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    logger = logging.getLogger(__name__)

    out_dir = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Set default values
    fixed_point_radius_threshold = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    # Load and combine dataframes for selected datasets
    statistics_manifest = load_dataframe_manifest(FIRST_PASSAGE_TIME_STATISTICS_MANIFEST_NAME)
    parameter_sweep_manifest = load_dataframe_manifest(
        FIRST_PASSAGE_TIME_PARAMETER_SWEEP_MANIFEST_NAME
    )

    fpt_stats_df = pd.concat(
        [
            load_dataframe(get_dataframe_location_for_dataset(statistics_manifest, dataset))
            for dataset in dataset_names
        ]
    )
    parameter_sweep_df = pd.concat(
        [
            load_dataframe(get_dataframe_location_for_dataset(parameter_sweep_manifest, dataset))
            for dataset in dataset_names
        ]
    )

    # report the mean and std of the per-bin means, and the median and std of
    # the per-bin medians, for each fixed point in each dataset, with flat
    # column names formatted as "<column>_<statistic> (<unit>)"; first passage
    # times are computed from `Column.SegData.TIME_HRS`, so they share its unit
    fpt_unit = COLUMN_METADATA[Column.SegData.TIME_HRS].unit
    stats_per_metric = {"mean": ["mean", "std"], "50%": ["median", "std"]}
    stats_per_col: dict[str, list[Any]] = {
        ColumnTemplate.FIRST_PASSAGE_TIME_METRIC % (metric, patch_type): stats
        for patch_type in ("grid_based", "cell_centered")
        for metric, stats in stats_per_metric.items()
    }
    fixed_point_groups = fpt_stats_df.groupby(
        [Column.DATASET, Column.VectorField.FIXED_POINT_INDEX, Column.FIXED_POINT_STABILITY]
    )
    mean_fpt_per_fp = fixed_point_groups.agg(stats_per_col)
    mean_fpt_per_fp.columns = [
        f"{col}_{stat} ({fpt_unit})" for col, stat in mean_fpt_per_fp.columns.to_flat_index()
    ]

    # fixed point locations should be the same for every bin of a given fixed
    # point, so take the single unique value in each group (`.item()` raises
    # an error if there is more than one)
    fixed_point_location_cols = [ColumnTemplate.FIXED_POINT % col for col in DYNAMICS_COLUMN_NAMES]
    fixed_point_locations = fixed_point_groups[fixed_point_location_cols].agg(
        lambda values: values.unique().item()
    )
    mean_fpt_per_fp = fixed_point_locations.join(mean_fpt_per_fp).reset_index()

    # map each dataset to its shear stress bin(s) and replicate number and add
    # them right after the dataset column; datasets with multiple flow
    # conditions (e.g., flow switch) list all bins, matching the dataset
    # summary tables
    dataset_configs = {
        dataset: load_dataset_config(dataset)
        for dataset in mean_fpt_per_fp[Column.DATASET].unique()
    }
    dataset_info_columns: dict[str, dict[str, str | int | None]] = {
        Column.SHEAR_STRESS_BIN: {
            dataset: ", ".join(str(fc.shear_stress_bin) for fc in config.flow_conditions)
            for dataset, config in dataset_configs.items()
        },
        Column.REPLICATE: {
            dataset: config.replicate_number for dataset, config in dataset_configs.items()
        },
    }
    insert_loc = mean_fpt_per_fp.columns.tolist().index(Column.DATASET) + 1
    for offset, (column, value_per_dataset) in enumerate(dataset_info_columns.items()):
        mean_fpt_per_fp.insert(
            loc=insert_loc + offset,
            column=column,
            value=[value_per_dataset[dataset] for dataset in mean_fpt_per_fp[Column.DATASET]],
        )

    # sort rows by shear stress bin, then replicate number; shear stress bins
    # are strings (e.g., "6, 24" for flow switch datasets), so compare them as
    # tuples of integers to sort naturally ("6" before "12"); dataset breaks
    # ties between datasets with the same bin and replicate number
    mean_fpt_per_fp = mean_fpt_per_fp.sort_values(
        by=[
            Column.SHEAR_STRESS_BIN,
            Column.REPLICATE,
            Column.DATASET,
            Column.VectorField.FIXED_POINT_INDEX,
        ],
        key=lambda col: (
            col.map(lambda bins: tuple(int(b) for b in bins.split(", ")))
            if col.name == Column.SHEAR_STRESS_BIN
            else col
        ),
        ignore_index=True,
    )
    mean_fpt_per_fp.to_csv(out_dir / "summary_stats_mfpt_per_fp.csv", index=False)

    # compute correlation between grid and tracked first passage time
    # statistics for each dataset and fixed point and save results as a
    # dataframe for plotting
    for metric_to_plot in ["mean", "median"]:

        # filter out nans and bins with too few trajectories for a certain measure
        # (either mean or median) for the correlation and line fitting steps
        fpt_stats_df_no_nan = filter_first_passage_time_by_min_num_trajectories(
            fpt_stats_df=fpt_stats_df,
            min_num_traj_per_bin=min_num_traj_per_bin,
            metric_for_filter=metric_to_plot,
        )
        # fit a line to the correlation between grid and tracked first passage
        # time statistics for each fixed point and dataset
        line_fit_df = build_first_passage_time_line_fit_results_dataframe(
            fpt_stats_df_no_nan=fpt_stats_df_no_nan,
            metric_to_fit=metric_to_plot,
        )
        # plot the correlation and parameter sweep results for each fixed point
        # (and add the figure examples to their own folder)
        for (dataset_name, fp_idx, fp_stability), grp_df in fpt_stats_df_no_nan.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.FIXED_POINT_STABILITY,
            ]
        ):
            fp_idx = round(fp_idx)
            out_dir_dataset = out_dir / dataset_name
            out_dir_dataset.mkdir(parents=True, exist_ok=True)

            # extract the line fit results for this dataset and fixed point
            line_fit_result = line_fit_df[
                (line_fit_df[Column.DATASET] == dataset_name)
                & (line_fit_df[Column.VectorField.FIXED_POINT_INDEX] == fp_idx)
            ]

            # plot the correlation results for this fixed point
            _ = plot_first_passage_time_correlations(
                dataset_name=dataset_name,
                first_passage_time_stats_df=grp_df,
                line_fit_df=line_fit_result,
                fixed_point_id=fp_idx,
                fixed_point_stability=fp_stability,
                out_dir=out_dir_dataset,
                metric_to_plot=metric_to_plot,
            )

            # histograms don't really work for 4D data (theta, r, rho, and FPT ratio),
            # so we will use a 3D scatter with color-coded points instead
            # if one of the columns is not being collapsed
            plot_first_passage_time_3d_scatter(
                fixed_point_id=fp_idx,
                dataset_name=dataset_name,
                fixed_point_stability=fp_stability,
                first_passage_time_df=grp_df,
                metric_to_plot=metric_to_plot,
                out_dir=out_dir_dataset,
            )
            # plot KDE of the first passage times for all of the bins thrown together
            plot_first_passage_time_histogram(
                dataset_name=dataset_name,
                fixed_point_id=fp_idx,
                fixed_point_stability=fp_stability,
                first_passage_time_df=grp_df,
                metric_to_plot=metric_to_plot,
                bin_width_for_hist=None,
                out_dir=out_dir_dataset,
            )
            # plot histograms of the numbers of trajectories per bin
            plot_first_passage_time_histogram(
                dataset_name=dataset_name,
                fixed_point_id=fp_idx,
                fixed_point_stability=fp_stability,
                first_passage_time_df=grp_df,
                metric_to_plot="count",
                bin_width_for_hist=1,
                out_dir=out_dir_dataset,
            )

        # plot the parameter sweep results (and add the figure examples to their own folder)
        for (dataset_name, fp_idx, fp_stability), grp_df in parameter_sweep_df.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.FIXED_POINT_STABILITY,
            ]
        ):
            fp_idx = round(fp_idx)
            out_dir_dataset = out_dir / dataset_name
            out_dir_dataset.mkdir(parents=True, exist_ok=True)

            _ = plot_first_passage_time_parameter_sweep(
                dataset_name=dataset_name,
                fixed_point_index=fp_idx,
                fixed_point_stability=fp_stability,
                first_passage_time_param_sweep_df=grp_df,
                fixed_point_radius_threshold=fixed_point_radius_threshold,
                out_dir=out_dir_dataset,
                metric_to_plot=metric_to_plot,
            )

        if len(dataset_names) > 2:
            # make a summary plot in both the regular output folder and also the figure output folder
            filename = f"FPT_correlation_summary_{metric_to_plot}"
            plot_first_passage_time_correlation_summary(line_fit_df, out_dir, filename)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
