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
        plot_first_passage_time_correlation_summary,
        plot_first_passage_time_correlations,
        plot_first_passage_time_parameter_sweep,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS

    # import odrpack

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

        # NOTE START OF ROUGH

        # dataset_name = dataset_config.name
        # time_units = TIME_STEP_IN_HOURS  # convert timeframes to hours

        for metric_to_plot in ["mean", "median"]:
            # NOTE MAKE LINE FIT DF
            # NOTE PLOT LINE FIT RESULTS
            # NOTE PLOT PARAM SWEEP RESULTS

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

            # convert the FPT (which is in timepoints) to physical units
            # most but not all of the columns are based on time in `first_passage_time_df`
            not_time_columns = [
                f"count{suffix}_grid",
                f"count{suffix}_tracked",
                Column.VectorField.BIN_INDEX,
                Column.VectorField.BIN_CENTER,
                Column.VectorField.BIN_EDGES,
            ]
            # the time columns are the set of columns in the dataframe that are not in
            # the not_time_columns list
            time_cols = list(set(fpt_stats_df_no_nan.columns) - set(not_time_columns))

            # # now we can convert all those time columns from timepoints to physical units
            # fpt_stats_df_no_nan[time_cols] *= time_units

            # do a linear regression to see if the FPTs from the tracked and grid trajectories
            # correlate depending on where they are in binned feature space
            # linemodel = LinearRegression()
            # linemodel.fit(
            #     fpt_stats_df_no_nan[[f"{metric}_grid"]],
            #     fpt_stats_df_no_nan[f"{metric}_tracked"],
            #     sample_weight=fpt_stats_df_no_nan[f"std_"],
            #     # sample_weight=fpt_stats_df_no_nan[Column.VectorField.BIN_COUNT],
            # )
            line_fit_df = (
                fpt_stats_df_no_nan.groupby([Column.DATASET, Column.VectorField.FIXED_POINT_INDEX])
                .apply(
                    lambda df: pd.Series(
                        index=[
                            [
                                "slope",
                                "intercept",
                                "r_value",
                                "p_value",
                                "std_err",
                            ]
                        ],
                        data=linregress(
                            x=df[f"{metric}_grid"],
                            y=df[f"{metric}_tracked"],
                        ),
                    )
                )
                .reset_index()
            )

            num_bins_series = fpt_stats_df_no_nan.groupby(
                [Column.DATASET, Column.VectorField.FIXED_POINT_INDEX]
            )[Column.VectorField.BIN_INDEX].nunique()

            for nm, grp_df in line_fit_df.groupby(
                [Column.DATASET, Column.VectorField.FIXED_POINT_INDEX]
            ):
                dataset_name, fp_idx = nm
                fixed_point_stability = fpt_stats_df_no_nan[
                    (fpt_stats_df_no_nan[Column.DATASET] == dataset_name)
                    & (fpt_stats_df_no_nan[Column.VectorField.FIXED_POINT_INDEX] == fp_idx)
                ][Column.VectorField.FIXED_POINT_STABILITY].iloc[0]
                plot_first_passage_time_correlations(
                    dataset_name=dataset_name,
                    first_passage_time_stats_df=grp_df,
                    fixed_point_index=fp_idx,
                    fixed_point_stability=fixed_point_stability,
                    out_dir=out_dir_figure,
                )
                plot_first_passage_time_parameter_sweep(
                    dataset_name=dataset_name,
                    parameter_sweep_df=parameter_sweep_df_no_nan,
                    fixed_point_index=fp_idx,
                    fixed_point_stability=fixed_point_stability,
                    out_dir=out_dir_figure,
                )
                # recommend to use orthogonal distance regression (ODR)
                # instead of ordinary least squares linear regression
                # TODO IMPLEMENT ODR INSTEAD OF OLS

                # NOTE END OF ROUGH

    # flatten the list of results and convert to a dataframe
    line_fit_results = [item for sublist in results for item in sublist]
    first_passage_time_correlation_summary_df = pd.DataFrame(line_fit_results)
    # we're only going to plot correlation results from the comparisons of the
    # first passage time means per bin
    first_passage_time_correlation_summary_df = first_passage_time_correlation_summary_df[
        first_passage_time_correlation_summary_df[Column.VectorField.FPT_METRIC] == "mean"
    ]

    out_dir_figure = out_dir / "for_figure"
    out_dir_figure.mkdir(parents=True, exist_ok=True)
    # make a summary plot in both the regular output folder and also the figure output folder
    for fdir in [out_dir, out_dir_figure]:
        plot_first_passage_time_correlation_summary(first_passage_time_correlation_summary_df, fdir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
