from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None):
    """
    Summarize segmentation counts across select datasets.

    #cdh5-segmentation #cdh5-tracking #nuclei-prediction #test-ready

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe summarize-segmentation-counts -d
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
    from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS,
        CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
        CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    # Create initial dictionary to keep track of segmentation counts (will later convert to DataFrame)
    seg_counts: dict[str, list] = {
        "dataset_name": [],
        "shear_stress_dyn/cm**2": [],
        "cell_line": [],
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
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_name_list = dataset_name_list[:1]

    for dataset_name in tqdm(dataset_name_list):
        # load the dataset config file to get some identifying information about the dataset
        dataset_config = load_dataset_config(dataset_name)
        shear_stress = [flow.shear_stress for flow in dataset_config.flow_conditions]
        cell_line = sequence_to_scalar(dataset_config.cell_lines)

        # load in segmentation features dataframe for the dataset if it was put through the classical
        # feature workflows, otherwise record "NaN" for the segmentation counts
        if dataset_name not in dataset_name_list_segmented:
            logging.info(
                f"Dataset {dataset_name} not found in live_cdh5_seg_based_feat_datasets collection, skipping."
            )
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
                Column.SegData.MAJOR_AXIS,
                Column.SegDataFilters.IS_INCLUDED,
            ]
            live_seg_feats_df = live_seg_feats_df_delayed[cols_to_compute].compute()

            # filter out rows based on automatic and manual timepoint annotations
            live_seg_feats_df = filter_dataframe_by_annotations(
                live_seg_feats_df,
                dataset_config,
                timepoint_annotations=ANNOTATIONS_TO_FILTER_OUT_FOR_SEGMENTATIONS,
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
    output_file = output_path / "segmentation_counts_across_datasets.tsv"
    seg_counts_df.to_csv(output_file, sep="\t", index=False)
    logger.info("Saved summary table to [ %s ]", output_file)

    major_axis_len_mean_all_px = seg_counts_df["major_axis_length_mean_px"].mean()
    major_axis_len_mean_all_um = seg_counts_df["major_axis_length_mean_um"].mean()
    cell_displacement_mean_all_px = seg_counts_df["cell_displacement_mean_px"].mean()

    print(f"Mean cell length across all datasets (px): {major_axis_len_mean_all_px}")
    print(f"Mean cell length across all datasets (um): {major_axis_len_mean_all_um}")
    print(f"Mean cell displacement across all datasets (px): {cell_displacement_mean_all_px}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
