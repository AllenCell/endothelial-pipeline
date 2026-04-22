"""This workflow computes the time of first passage for each track in the provided datasets."""

from typing import Literal

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    minimum_track_length: int | None = None,
    run_FPT_threshold_parameter_sweep: bool = True,
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
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.analyze.track_integration import (
        compute_and_plot_first_passage_times_one_dataset,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_first_passage_time_correlation_summary,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
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

    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futures = []
        for dataset_name in dataset_names:
            futures.append(
                executor.submit(
                    compute_and_plot_first_passage_times_one_dataset,
                    dataset_name=dataset_name,
                    out_dir=out_dir,
                    minimum_track_length=minimum_track_length,
                    run_FPT_threshold_parameter_sweep=run_FPT_threshold_parameter_sweep,
                    fixed_point_radius_threshold=fixed_point_radius_threshold,
                    min_num_traj_per_bin=min_num_traj_per_bin,
                    bin_size_theta_deg=bin_size_theta_deg,
                    bin_size_radius=bin_size_radius,
                    bin_size_rho=bin_size_rho,
                    collapse_feature=collapse_feature,
                )
            )
        results = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing FPT for datasets",
        ):
            results.append(future.result())

    # flatten the list of results and convert to a dataframe
    line_fit_results = [item for sublist in results for item in sublist]
    first_passage_time_correlation_summary_df = pd.DataFrame(line_fit_results)
    # we're only going to plot correlation results from the comparisons of the
    # first passage time means per bin
    first_passage_time_correlation_summary_df = first_passage_time_correlation_summary_df[
        first_passage_time_correlation_summary_df[Column.VectorField.FPT_METRIC] == "mean"
    ]
    plot_first_passage_time_correlation_summary(first_passage_time_correlation_summary_df, out_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
