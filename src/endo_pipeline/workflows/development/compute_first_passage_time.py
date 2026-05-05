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
    n_proc: int = 6,
    upload_to_fms: bool = False,
) -> None:

    import logging
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import (
        build_fms_annotations,
        get_output_path,
        make_name_unique,
        upload_file_to_fms,
    )
    from endo_pipeline.library.analyze.track_integration import (
        build_fpt_line_fit_results_df,
        compute_first_passage_times_one_dataset,
        filter_fpt_stats_df_by_min_num_trajectories,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_first_passage_time_3d_scatter,
        plot_first_passage_time_correlation_summary,
        plot_first_passage_time_correlations,
        plot_first_passage_time_histogram,
        plot_first_passage_time_parameter_sweep,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        build_dataframe_location_from_path,
        create_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import LONG_TRACK_THRESHOLD_LENGTH
    from endo_pipeline.settings.examples import FPT_FIG_EXAMPLES
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FIRST_PASSAGE_TIME_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    dataset_names = datasets or SUMMARY_PLOT_DATASETS["intermediate"]

    if minimum_track_length is None:
        minimum_track_length = LONG_TRACK_THRESHOLD_LENGTH

    if fixed_point_radius_threshold is None:
        fixed_point_radius_threshold = MIGRATION_COHERENCE_COLORMAP_BIN_SIZE

    if DEMO_MODE:
        fpt_fig_example_datasets = {ex.dataset_name for ex in FPT_FIG_EXAMPLES.values()}
        dataset_names = list(fpt_fig_example_datasets | set(SUMMARY_PLOT_DATASETS["low_high"]))
        logger.info(f"Running in demo mode, only processing datasets [ {dataset_names} ]")
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

    # collect the FPT statistics and parameter sweep results into dataframes
    fpt_stats_df_list, parameter_sweep_df_list = zip(*results, strict=True)
    fpt_stats_df_list = [df_one_fp for df_list in fpt_stats_df_list for df_one_fp in df_list]
    parameter_sweep_df_list = [
        df_one_fp for df_list in parameter_sweep_df_list for df_one_fp in df_list
    ]
    fpt_stats_df = pd.concat(fpt_stats_df_list, ignore_index=True)
    parameter_sweep_df = pd.concat(parameter_sweep_df_list, ignore_index=True)

    # save the first passage time statistics and parameter sweep results as dataframes for plotting
    fpt_stats_df_savepath = make_name_unique(out_dir / "first_passage_time_stats.parquet")
    fpt_stats_df.to_parquet(fpt_stats_df_savepath)
    fpt_param_sweep_df_savepath = make_name_unique(
        out_dir / "first_passage_time_parameter_sweep.parquet"
    )
    parameter_sweep_df.to_parquet(fpt_param_sweep_df_savepath)

    # If upload_to_fms is True, upload the parquet file to FMS and
    # register the FMS ID in the manifest; otherwise, register the local
    # path in the manifest
    fpt_manifest = create_dataframe_manifest(
        FIRST_PASSAGE_TIME_MANIFEST_NAME, workflow_name=__file__
    )
    logger.info("First passage time dataframes will be saved to: [ %s ]", out_dir)

    if upload_to_fms and not DEMO_MODE:
        dataset_configs = [load_dataset_config(name) for name in dataset_names]
        model_manifest = load_model_manifest(manifest_name=DEFAULT_MODEL_MANIFEST_NAME)
        fms_annotations = build_fms_annotations(
            dataset=dataset_configs,
            model_manifest=model_manifest,
            run_name=DEFAULT_MODEL_RUN_NAME,
            additional_notes="First passage time statistics and parameter sweep dataframes for integration analysis.",
        )
        fms_id_fpt_stats = upload_file_to_fms(fpt_stats_df_savepath, fms_annotations, "parquet")
        fms_id_fpt_param_sweep = upload_file_to_fms(
            fpt_param_sweep_df_savepath, fms_annotations, "parquet"
        )
        logger.info(
            "Uploaded first passage time statistics table to FMS with ID [ %s ]", fms_id_fpt_stats
        )
        logger.info(
            "Uploaded first passage time parameter sweep table to FMS with ID [ %s ]",
            fms_id_fpt_param_sweep,
        )
        fpt_manifest.locations["first_passage_time_statistics"] = DataframeLocation(
            fmsid=fms_id_fpt_stats
        )
        fpt_manifest.locations["first_passage_time_parameter_sweep"] = DataframeLocation(
            fmsid=fms_id_fpt_param_sweep
        )
    else:
        stats_loc = fpt_manifest.locations.get("first_passage_time_statistics")
        if stats_loc is None or stats_loc.fmsid is None:
            fpt_manifest.locations["first_passage_time_statistics"] = (
                build_dataframe_location_from_path(fpt_stats_df_savepath)
            )
            fpt_manifest.locations["first_passage_time_parameter_sweep"] = (
                build_dataframe_location_from_path(fpt_param_sweep_df_savepath)
            )
    # register the parameters that the worfklow was run with in the manifest for
    # provenance tracking and reproducibility, even if not uploading to FMS
    fpt_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "datasets": dataset_names,
        "fixed_point_radius_threshold": fixed_point_radius_threshold,
        "minimum_track_length": minimum_track_length,
        "min_num_traj_per_bin": min_num_traj_per_bin,
        "bin_size_theta_deg": bin_size_theta_deg,
        "bin_size_radius": bin_size_radius,
        "bin_size_rho": bin_size_rho,
        "collapse_feature": collapse_feature,
        "n_proc": n_proc,
    }
    save_dataframe_manifest(fpt_manifest)

    # then compute correlation between grid and tracked first passage time
    # statistics for each dataset and fixed point and save results as a
    # dataframe for plotting
    for metric_to_plot in ["mean", "median"]:

        # filter out nans and bins with too few trajectories for a certain measure
        # (either mean or median) for the correlation and line fitting steps
        fpt_stats_df_no_nan = filter_fpt_stats_df_by_min_num_trajectories(
            fpt_stats_df=fpt_stats_df,
            min_num_traj_per_bin=min_num_traj_per_bin,
            metric_for_filter=metric_to_plot,
        )
        # fit a line to the correlation between grid and tracked first passage
        # time statistics for each fixed point and dataset
        line_fit_df = build_fpt_line_fit_results_df(
            fpt_stats_df_no_nan=fpt_stats_df_no_nan,
            metric_to_fit=metric_to_plot,
        )
        # plot the correlation and parameter sweep results for each fixed point
        # (and add the figure examples to their own folder)
        for (dataset_name, fp_idx, fp_stability), grp_df in fpt_stats_df_no_nan.groupby(
            [
                Column.DATASET,
                Column.VectorField.FIXED_POINT_INDEX,
                Column.VectorField.STABILITY,
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
                Column.VectorField.STABILITY,
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
