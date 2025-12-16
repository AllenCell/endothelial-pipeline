"""This script makes a table of segmentation counts across all live 20X 48hr timelapse datasets."""


def main():
    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
    from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME, ColumnName

    # Create initial dictionary to keep track of segmentation counts (will later convert to DataFrame)
    seg_counts: dict[str, list] = {
        "dataset_name": [],
        "shear_stress_dyn/cm**2": [],
        "cell_line": [],
        "num_nuclei_predictions": [],
        "num_cell_segmentations_before_filter": [],
        "num_cell_segmentations_after_filter": [],
    }

    # generate sequence of unique datasets to process and add to the seg_counts dictionary
    dataset_name_list_segmented = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")
    dataset_name_list = set(
        dataset_name_list_segmented + get_datasets_in_collection("perturbation")
    )

    for dataset_name in dataset_name_list:
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
            num_nuc_pred = np.nan
            num_cell_seg_before_filt = np.nan
            num_cell_seg_after_filt = np.nan
        else:
            # load segmentation features dataframe
            live_seg_manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
            live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
            live_seg_feats_df = load_dataframe(live_seg_location)

            # segmentation counts recorded in the table were done at each timepoint
            # (a.k.a. the image_index) for one position at a time, therefore we need
            # to do a groubpy on both image_index and position before summing the
            # totals in each dataset across all timepoints and positions
            num_nuc_pred = (
                live_seg_feats_df.groupby(["image_index", ColumnName.POSITION])
                .total_nuclei_count_at_T.apply(sequence_to_scalar)
                .sum()
            )
            num_cell_seg_before_filt = (
                live_seg_feats_df.groupby(["image_index", ColumnName.POSITION])
                .num_unique_tracks_before_filtering_at_T.apply(sequence_to_scalar)
                .sum()
            )

            # filter out rows based on automatic and manual timepoint annotations
            live_seg_feats_df["dataset"] = live_seg_feats_df["dataset_name"]
            live_seg_feats_df["frame_number"] = live_seg_feats_df["image_index"]
            annotations_to_filter_out = [
                TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
                TimepointAnnotation.GFP_SCOPE_ERROR,
            ]
            live_seg_feats_df = filter_dataframe_by_annotations(
                live_seg_feats_df, dataset_config, timepoint_annotations=annotations_to_filter_out
            )

            # dropna is here to remove NaNs which will raise an error when trying to
            # extract a single number from the table at that timepoint and position
            # using the `sequence_to_scalar` function
            num_cell_seg_after_filt = int(
                live_seg_feats_df.dropna(
                    axis="index", subset=["num_unique_tracks_after_filtering_at_T"]
                )
                .groupby(["image_index", ColumnName.POSITION])
                .num_unique_tracks_after_filtering_at_T.apply(sequence_to_scalar)
                .sum()
            )
            # NOTE that num_cell_seg_before_filt can also be calculated with
            # live_seg_feats_df.groupby(["image_index", ColumnName.POSITION]).label.nunique().sum()
            # and num_cell_seg_after_filt with
            # live_seg_feats_df.query("is_included==True").groupby(["image_index", ColumnName.POSITION]).label.nunique().sum()

            # delete the segmentation features dataframe to keep memory usage down
            del live_seg_feats_df

        # add identifying information and segmentation counts to the seg_counts dictionary
        seg_counts["dataset_name"].append(dataset_name)
        seg_counts["shear_stress_dyn/cm**2"].append(shear_stress)
        seg_counts["cell_line"].append(cell_line)
        seg_counts["num_nuclei_predictions"].append(num_nuc_pred)
        seg_counts["num_cell_segmentations_before_filter"].append(num_cell_seg_before_filt)
        seg_counts["num_cell_segmentations_after_filter"].append(num_cell_seg_after_filt)

    # convert the seg_counts dictionary to a dataframe and save the results
    seg_counts_df = pd.DataFrame(seg_counts)
    out_dir = get_output_path(__file__)
    seg_counts_df.to_csv(out_dir / "segmentation_counts_across_datasets.tsv", sep="\t", index=False)


if __name__ == "__main__":
    from endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    ipython_cli_flexecute(main)
