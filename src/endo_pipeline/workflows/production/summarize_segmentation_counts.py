from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None):
    """
    Summarize segmentation counts across select datasets.

    #cdh5-segmentation #cdh5-tracking #nuclei-prediction

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe summarize-segmentation-counts -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe summarize-segmentation-counts --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will summarize
    only the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to summarize.
    """

    import logging

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
    from endo_pipeline.library.model.train_model import get_included_frames_for_model
    from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS,
        CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
        CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    # Create initial dictionary to keep track of segmentation counts (will later convert to DataFrame)
    seg_counts: dict[str, list] = {
        "dataset_name": [],
        "shear_stress_dyn/cm**2": [],
        "cell_line": [],
        "num_nuclei_predictions": [],
        "num_cell_segmentations_before_filter": [],
        "num_cell_segmentations_after_filter": [],
        "num_tracks_before_filter": [],
        "num_tracks_after_filter": [],
        "dataset_duration_timeframes": [],
        "num_timeframes_left_after_filter": [],
        "num_timeframes_for_training": [],
        "major_axis_length_mean_um": [],
        "major_axis_length_std_um": [],
        "major_axis_length_median_um": [],
        "major_axis_length_mean_px": [],
        "major_axis_length_std_px": [],
        "major_axis_length_median_px": [],
        "cell_displacement_mean_px": [],
        "cell_displacement_median_px": [],
    }

    # generate sequence of unique datasets to process and add to the seg_counts dictionary
    dataset_name_list_segmented = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")
    dataset_name_list = datasets or list(
        set(dataset_name_list_segmented + get_datasets_in_collection("perturbation"))
    )

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_name_list = dataset_name_list[:1]

    for dataset_name in tqdm(dataset_name_list):
        # load the dataset config file to get some identifying information about the dataset
        dataset_config = load_dataset_config(dataset_name)
        shear_stress = [flow.shear_stress for flow in dataset_config.flow_conditions]
        cell_line = sequence_to_scalar(dataset_config.cell_lines)

        # get list of timepoints that were included for training for each position
        only_include_frames = get_included_frames_for_model(dataset_config)
        num_timepoints_for_training = sum(
            [len(only_include_frames[pos]) for pos in only_include_frames]
        )

        # load in segmentation features dataframe for the dataset if it was put through the classical
        # feature workflows, otherwise record "NaN" for the segmentation counts
        if dataset_name not in dataset_name_list_segmented:
            logging.info(
                f"Dataset {dataset_name} not found in live_cdh5_seg_based_feat_datasets collection, skipping."
            )
            num_nuc_pred = np.nan
            num_cell_seg_before_filt = np.nan
            num_cell_seg_after_filt = np.nan
            num_tracks_before_filt = np.nan
            num_tracks_left_after_filter = np.nan
            num_timepoints_left_after_filter = np.nan
            seg_lengths_px_mean = np.nan
            seg_lengths_px_std = np.nan
            seg_lengths_px_median = np.nan
            seg_lengths_um_mean = np.nan
            seg_lengths_um_std = np.nan
            seg_lengths_um_median = np.nan
            cell_seg_displacement_mean = np.nan
            cell_seg_displacement_median = np.nan
        else:
            # load segmentation features dataframe
            live_seg_manifest = load_dataframe_manifest(
                CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME
            )
            live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
            live_seg_feats_df_delayed = load_dataframe(live_seg_location, delay=True)
            cols_to_compute = [
                Column.DATASET,
                Column.POSITION,
                Column.TIMEPOINT,
                Column.TRACK_ID,
                Column.SegData.NUM_NUCLEI_AT_TIMEPOINT,
                Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
                Column.SegData.NUM_TRACKS_AFTER_FILTERING,
                Column.SegData.MAJOR_AXIS,
                Column.SegDataFilters.IS_INCLUDED,
            ]
            live_seg_feats_df = live_seg_feats_df_delayed[cols_to_compute].compute()

            # segmentation counts recorded in the table were done at each timepoint
            # (a.k.a. the image_index) for one position at a time, therefore we need
            # to do a groubpy on both image_index and position before summing the
            # totals in each dataset across all timepoints and positions
            num_nuc_pred = (
                live_seg_feats_df.groupby([Column.TIMEPOINT, Column.POSITION])[
                    Column.SegData.NUM_NUCLEI_AT_TIMEPOINT
                ]
                .apply(sequence_to_scalar)
                .sum()
            )
            num_cell_seg_before_filt = (
                live_seg_feats_df.groupby([Column.TIMEPOINT, Column.POSITION])[
                    Column.SegData.NUM_TRACKS_BEFORE_FILTERING
                ]
                .apply(sequence_to_scalar)
                .sum()
            )
            num_tracks_before_filt = (
                live_seg_feats_df.groupby(Column.POSITION)[Column.TRACK_ID].nunique().sum()
            )

            # filter out rows based on automatic and manual timepoint annotations
            live_seg_feats_df = filter_dataframe_by_annotations(
                live_seg_feats_df,
                dataset_config,
                timepoint_annotations=ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS,
            )

            # dropna is here to remove NaNs which will raise an error when trying to
            # extract a single number from the table at that timepoint and position
            # using the `sequence_to_scalar` function
            num_cell_seg_after_filt = int(
                live_seg_feats_df.dropna(
                    axis="index", subset=[Column.SegData.NUM_TRACKS_AFTER_FILTERING]
                )
                .groupby([Column.TIMEPOINT, Column.POSITION])[
                    Column.SegData.NUM_TRACKS_AFTER_FILTERING
                ]
                .apply(sequence_to_scalar)
                .sum()
            )

            # add number of timepoints left after filtering to the dataset
            # the "is_included" column in the dataframe is defined when the dataframe is constructed
            # based on whether the track at that timepoint passed all filtering steps or not
            # (see endo_pipeline\library\analyze\live_data_manifest\lib_make_seg_feats_manifest.add_filter_columns for details)
            num_timepoints_left_after_filter = (
                live_seg_feats_df[live_seg_feats_df[Column.SegDataFilters.IS_INCLUDED]]
                .groupby([Column.POSITION])[Column.TIMEPOINT]
                .nunique()
                .sum()
            )
            num_tracks_left_after_filter = (
                live_seg_feats_df[live_seg_feats_df[Column.SegDataFilters.IS_INCLUDED]]
                .groupby(Column.POSITION)[Column.TRACK_ID]
                .nunique()
                .sum()
            )

            # add some descriptive statistics about cell lengths to the dataset
            # (this is approximated by reporting the major axis of an ellipse fit to a segmentation)
            seg_lengths_px_mean = live_seg_feats_df[Column.SegData.MAJOR_AXIS].mean()
            seg_lengths_px_std = live_seg_feats_df[Column.SegData.MAJOR_AXIS].std()
            seg_lengths_px_median = live_seg_feats_df[Column.SegData.MAJOR_AXIS].median()
            seg_lengths_um_mean = (
                live_seg_feats_df[Column.SegData.MAJOR_AXIS].mean()
                * dataset_config.pixel_size_xy_in_um
            )
            seg_lengths_um_std = (
                live_seg_feats_df[Column.SegData.MAJOR_AXIS].std()
                * dataset_config.pixel_size_xy_in_um
            )
            seg_lengths_um_median = (
                live_seg_feats_df[Column.SegData.MAJOR_AXIS].median()
                * dataset_config.pixel_size_xy_in_um
            )

            # delete the segmentation features dataframe to keep memory usage down
            del live_seg_feats_df

            live_seg_filtered_manifest = load_dataframe_manifest(
                CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME
            )
            live_seg_filtered_location = get_dataframe_location_for_dataset(
                live_seg_filtered_manifest, dataset_name
            )
            live_seg_feats_filtered_df_delayed = load_dataframe(
                live_seg_filtered_location, delay=True
            )
            cols_to_compute_filtered = [
                Column.DATASET,
                Column.POSITION,
                Column.TIMEPOINT,
                Column.TRACK_ID,
                Column.SegData.CENTROID_VELOCITY_UM_PER_MIN,
            ]
            live_seg_feats_filtered_df = live_seg_feats_filtered_df_delayed[
                cols_to_compute_filtered
            ].compute()

            cell_seg_speed_mean = (
                live_seg_feats_filtered_df[Column.SegData.CENTROID_VELOCITY_UM_PER_MIN]
                .dropna()
                .mean()
            )
            cell_seg_displacement_mean = (
                dataset_config.time_interval_in_minutes
                / dataset_config.pixel_size_xy_in_um
                * cell_seg_speed_mean
            )
            cell_seg_speed_median = (
                live_seg_feats_filtered_df[Column.SegData.CENTROID_VELOCITY_UM_PER_MIN]
                .dropna()
                .median()
            )
            cell_seg_displacement_median = (
                dataset_config.time_interval_in_minutes
                / dataset_config.pixel_size_xy_in_um
                * cell_seg_speed_median
            )

        # add identifying information and segmentation counts to the seg_counts dictionary
        seg_counts["dataset_name"].append(dataset_name.split("_")[0])
        seg_counts["shear_stress_dyn/cm**2"].append(shear_stress)
        seg_counts["cell_line"].append(cell_line)
        # add segmentation counts information
        seg_counts["num_nuclei_predictions"].append(num_nuc_pred)
        seg_counts["num_cell_segmentations_before_filter"].append(num_cell_seg_before_filt)
        seg_counts["num_cell_segmentations_after_filter"].append(num_cell_seg_after_filt)
        seg_counts["num_tracks_before_filter"].append(num_tracks_before_filt)
        seg_counts["num_tracks_after_filter"].append(num_tracks_left_after_filter)
        # add dataset duration information
        seg_counts["dataset_duration_timeframes"].append(dataset_config.duration)
        seg_counts["num_timeframes_left_after_filter"].append(num_timepoints_left_after_filter)
        seg_counts["num_timeframes_for_training"].append(num_timepoints_for_training)
        # add the cell length statistics
        seg_counts["major_axis_length_mean_px"].append(seg_lengths_px_mean)
        seg_counts["major_axis_length_std_px"].append(seg_lengths_px_std)
        seg_counts["major_axis_length_median_px"].append(seg_lengths_px_median)
        seg_counts["major_axis_length_mean_um"].append(seg_lengths_um_mean)
        seg_counts["major_axis_length_std_um"].append(seg_lengths_um_std)
        seg_counts["major_axis_length_median_um"].append(seg_lengths_um_median)
        # add the cell displacements (in pixels) between timepoints based on the segmentations
        seg_counts["cell_displacement_mean_px"].append(cell_seg_displacement_mean)
        seg_counts["cell_displacement_median_px"].append(cell_seg_displacement_median)

    # convert the seg_counts dictionary to a dataframe and save the results
    seg_counts_df = pd.DataFrame(seg_counts)
    out_dir = get_output_path(__file__)
    seg_counts_df.to_csv(out_dir / "segmentation_counts_across_datasets.tsv", sep="\t", index=False)

    major_axis_len_mean_all_px = seg_counts_df["major_axis_length_mean_px"].mean()
    major_axis_len_mean_all_um = seg_counts_df["major_axis_length_mean_um"].mean()

    logger.info(f"Mean cell length across all datasets (px): {major_axis_len_mean_all_px}")
    logger.info(f"Mean cell length across all datasets (um): {major_axis_len_mean_all_um}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
