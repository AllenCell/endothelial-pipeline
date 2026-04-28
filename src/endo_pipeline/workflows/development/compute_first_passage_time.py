"""This workflow computes the time of first passage for each track in the provided datasets."""

from typing import Literal

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    minimum_track_length: int | None = None,
    fixed_point_radius_threshold: float | None = None,
    min_num_traj_per_bin: int = 10,
    bin_size_theta_deg: float | None = None,
    bin_size_radius: float | None = None,
    bin_size_rho: float | None = None,
    collapse_feature: Literal["theta", "radius", "rho"] | None = None,
    n_proc: int = 4,
) -> None:

    import logging
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd
    from scipy.stats import linregress
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.analyze.track_integration import (
        compute_first_passage_times_one_dataset,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_first_passage_time_3d_scatter,
        plot_first_passage_time_correlation_summary,
        plot_first_passage_time_correlations,
        plot_first_passage_time_histogram,
        plot_first_passage_time_parameter_sweep,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
    from endo_pipeline.settings.examples import FPT_FIG_EXAMPLES
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS

    logger = logging.getLogger(__name__)

    dataset_names = datasets or SUMMARY_PLOT_DATASETS["intermediate"]

    if minimum_track_length is None:
        minimum_track_length = LONG_TRACK_THRESHOLD_LENGTH

    if fixed_point_radius_threshold is None:
        fixed_point_radius_threshold = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    if DEMO_MODE:
        dataset_names = dataset_names[:3]
        logger.info(f"Running in demo mode, processing only the first 3 datasets: {dataset_names}")
        out_dir = get_output_path(__file__, "demo")
    else:
        out_dir = get_output_path(__file__)

    out_dir_figure = out_dir / "for_figure"
    out_dir_figure.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=min(n_proc, len(dataset_names))) as executor:
        futures: list = []
        for dataset_name in dataset_names:
            futures.append(
                executor.submit(
                    compute_first_passage_times_one_dataset,
                    dataset_name=dataset_name,
                    minimum_track_length=minimum_track_length,
                    fixed_point_radius_threshold=fixed_point_radius_threshold,
                    bin_size_theta_deg=bin_size_theta_deg,
                    bin_size_radius=bin_size_radius,
                    bin_size_rho=bin_size_rho,
                    collapse_feature=collapse_feature,
                )
            )
        results: list = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing FPT for datasets",
        ):
            results.append(future.result())
        fpt_stats_df_list, parameter_sweep_df_list = zip(*results, strict=True)
        fpt_stats_df_list = [df_one_fp for df_list in fpt_stats_df_list for df_one_fp in df_list]
        parameter_sweep_df_list = [
            df_one_fp for df_list in parameter_sweep_df_list for df_one_fp in df_list
        ]
        fpt_stats_df = pd.concat(fpt_stats_df_list, ignore_index=True)
        parameter_sweep_df = pd.concat(parameter_sweep_df_list, ignore_index=True)

        # then compute correlation between grid and tracked first passage time
        # statistics for each dataset and fixed point and save results as a
        # dataframe for plotting
        for metric_to_plot in ["mean", "median"]:
            # the column title is "50%" for 50th percentile in `pd.describe`` instead of
            # mean so correct that if "median" was chosen
            metric = "50%" if metric_to_plot == "median" else metric_to_plot

            suffix = Column.VectorField.FIRST_PASSAGE_TIME_SUFFIX
            metric = f"{metric}{suffix}"

            # NaN values are unacceptable for the linear regression
            fpt_stats_df_no_nan = fpt_stats_df.copy().dropna(
                subset=[f"{metric}_grid", f"{metric}_tracked"]
            )
            # keep only the bins with the minimum number of tracks per bin in them
            fpt_stats_df_no_nan = fpt_stats_df_no_nan[
                fpt_stats_df_no_nan["count_first_passage_time_grid"] >= min_num_traj_per_bin
            ]
            fpt_stats_df_no_nan = fpt_stats_df_no_nan[
                fpt_stats_df_no_nan["count_first_passage_time_tracked"] >= min_num_traj_per_bin
            ]

            # do a linear regression to see if the FPTs from the tracked and grid trajectories
            # correlate depending on where they are in binned feature space
            line_fit_df = (
                fpt_stats_df_no_nan.groupby(
                    [
                        Column.DATASET,
                        Column.VectorField.FIXED_POINT_INDEX,
                        Column.VectorField.STABILITY,
                    ]
                )
                .apply(
                    lambda df, metric=metric: pd.Series(
                        index=[
                            "slope",
                            "intercept",
                            "r_value",
                            "p_value",
                            "std_err",
                        ],
                        data=linregress(
                            x=df[f"{metric}_grid"],
                            y=df[f"{metric}_tracked"],
                        ),
                    )
                )
                .reset_index()
            )

            # plot the correlation and parameter sweep results for each fixed point
            # (and add the figure examples to their own folder)
            for nm, grp_df in fpt_stats_df_no_nan.groupby(
                [
                    Column.DATASET,
                    Column.VectorField.FIXED_POINT_INDEX,
                    Column.VectorField.STABILITY,
                ]
            ):
                dataset_name, fp_idx, fp_stability = nm
                out_dir_dataset = out_dir_figure / dataset_name
                out_dir_dataset.mkdir(parents=True, exist_ok=True)

                # extract the line fit results for this dataset and fixed point
                line_fit_result = line_fit_df[
                    (line_fit_df[Column.DATASET] == dataset_name)
                    & (line_fit_df[Column.VectorField.FIXED_POINT_INDEX] == fp_idx)
                ]
                slope = line_fit_result["slope"].unique().item()
                intercept = line_fit_result["intercept"].unique().item()
                r_value = line_fit_result["r_value"].unique().item()

                # plot the correlation results for this fixed point
                plot_first_passage_time_correlations(
                    dataset_name=dataset_name,
                    first_passage_time_stats_df=grp_df,
                    fixed_point_id=fp_idx,
                    fixed_point_stability=fp_stability,
                    slope=slope,
                    intercept=intercept,
                    r_value=r_value,
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

                if dataset_name in FPT_FIG_EXAMPLES and metric_to_plot == "mean":
                    plot_first_passage_time_correlations(
                        dataset_name=dataset_name,
                        first_passage_time_stats_df=grp_df,
                        fixed_point_id=fp_idx,
                        fixed_point_stability=fp_stability,
                        slope=slope,
                        intercept=intercept,
                        r_value=r_value,
                        out_dir=out_dir_figure,
                        metric_to_plot=metric_to_plot,
                    )
            # plot the parameter sweep results (and add the figure examples to their own folder)
            for nm, grp_df in parameter_sweep_df.groupby(
                [
                    Column.DATASET,
                    Column.VectorField.FIXED_POINT_INDEX,
                    Column.VectorField.STABILITY,
                ]
            ):
                dataset_name, fp_idx, fp_stability = nm
                plot_first_passage_time_parameter_sweep(
                    dataset_name=dataset_name,
                    fixed_point_index=fp_idx,
                    fixed_point_stability=fp_stability,
                    first_passage_time_param_sweep_df=grp_df,
                    fixed_point_radius_threshold=fixed_point_radius_threshold,
                    out_dir=out_dir_dataset,
                    metric_to_plot=metric_to_plot,
                )
                if dataset_name in FPT_FIG_EXAMPLES and metric_to_plot == "mean":
                    plot_first_passage_time_parameter_sweep(
                        dataset_name=dataset_name,
                        fixed_point_index=fp_idx,
                        fixed_point_stability=fp_stability,
                        first_passage_time_param_sweep_df=grp_df,
                        fixed_point_radius_threshold=fixed_point_radius_threshold,
                        out_dir=out_dir_figure,
                        metric_to_plot=metric_to_plot,
                    )

            if len(dataset_names) > 1:
                # make a summary plot in both the regular output folder and also the figure output folder
                filename = f"FPT_correlation_summary_{metric_to_plot}"
                plot_first_passage_time_correlation_summary(line_fit_df, out_dir, filename)
                if metric_to_plot == "mean":
                    plot_first_passage_time_correlation_summary(
                        line_fit_df, out_dir_figure, filename
                    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
