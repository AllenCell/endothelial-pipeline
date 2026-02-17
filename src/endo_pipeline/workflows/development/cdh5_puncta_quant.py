# examples of endo cells with puncta
# dataset_name = "20250402_20X"
# timepoint = 166
# position = 0
# track_id (label_id)

# at rear:
# 775 (141)
# 4076 (343)
# 3647 (203)

# puncta are elsewhere:
# 1781 (213)
# 2964 (193)
# 3064 (47)


# examples of endo cell under high flow
# dataset_name = "20250611_20X"
# timepoint = 343
# position = 0
# 6342 (169)


def main(n_cores=1):
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from endo_pipeline.configs import get_subset_of_timepoint_annotations, load_dataset_config
    from endo_pipeline.configs.dataset_io import concatenate_and_save_feature_tables
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
    from endo_pipeline.library.analyze.intensity_features import (
        calculate_edge_intensity_distribution_for_segmentations_mp,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
        SEGMENTATION_FEATURE_COLUMNS,
    )

    low_flow_dataset_name = "20250402_20X"
    high_flow_dataset_name = "20250611_20X"
    interm_flow_dataset_name = "20250818_20X"
    dataset_name_list = [low_flow_dataset_name, high_flow_dataset_name, interm_flow_dataset_name]

    for dataset_name in dataset_name_list:
        out_dir = get_output_path(__file__)

        # analysis_queue = build_analysis_queue(
        #     dataset_name_list=[dataset_name],
        #     save_output=False,
        #     out_dir=out_dir,
        # )
        # seg_of_interest = 203

        # analysis_queue = build_analysis_queue(
        #     dataset_name_list=["20250611_20X"],
        #     save_output=False,
        #     out_dir=None,
        #     overwrite=False,
        #     verbose=True,
        #     image_validation_frequency=48,
        #     is_test=True,
        #     t_start=343,
        #     t_final=344,
        # )
        # seg_of_interest = 169

        # args = analysis_queue[0]
        # dataset_name = args["dataset_name"]
        # position = args["position"]
        # tp = args["T"]

        df_manifest = load_dataframe_manifest(DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME)
        df_loc = get_dataframe_location_for_dataset(df_manifest, dataset_name)
        df = load_dataframe(df_loc, delay=True)
        df = df.reset_index(drop=True)
        dataset_config = load_dataset_config(dataset_name)

        dataset_info_cols = [
            "dataset",
            "position",
            "image_index",
        ]
        crop_cols = [
            "start_x_cdh5_seg",
            "start_y_cdh5_seg",
            "end_x_cdh5_seg",
            "end_y_cdh5_seg",
        ]
        seg_info_cols = [
            "label",
            "track_id",
            "centroid_X",
            "centroid_Y",
        ]
        dynamics_cols = SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"]
        filter_cols = ["is_included"]
        cols_to_compute = list(
            set(dataset_info_cols + crop_cols + seg_info_cols + filter_cols + dynamics_cols)
        )
        df_subset = df[cols_to_compute].compute()

        df_subset = df_subset[df_subset.is_included]
        df_subset["frame_number"] = df_subset["image_index"]
        df_subset["dataset"] = dataset_name

        annotations_to_ignore: list = []
        timepoint_annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=annotations_to_ignore
        )
        df_subset = filter_dataframe_by_annotations(
            df_subset,
            dataset_config=dataset_config,
            timepoint_annotations=timepoint_annotations,
        )
        df_subset["output_dir"] = out_dir.as_posix()

        df_subset = df_subset.query("position == 0")

        grps = df_subset.groupby(["dataset", "position", "image_index"])

        worker_pool = ProcessPoolExecutor(max_workers=n_cores)

        list(
            tqdm(
                worker_pool.map(
                    calculate_edge_intensity_distribution_for_segmentations_mp,
                    grps,
                ),
                desc=f"Calculating edge intensity distributions for {dataset_name}",
                total=len(grps),
            )
        )

        concatenate_and_save_feature_tables(
            out_dir=out_dir,
            dataset_name=dataset_name,
            out_file_suffix="edge_intensities",
            input_filename_contains="edge_intensities",
            file_extension=".parquet",
            remove_initial_files_and_folders=True,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
