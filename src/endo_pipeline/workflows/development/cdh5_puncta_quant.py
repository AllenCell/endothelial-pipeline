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
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
    from endo_pipeline.library.analyze.integration.track_integration import (
        load_pc_diffae_liveseg_feats_merged_table,
    )
    from endo_pipeline.library.analyze.intensity_features import (
        calculate_edge_intensity_distribution_for_segmentations_mp,
    )
    from endo_pipeline.settings.workflow_defaults import SEGMENTATION_FEATURE_COLUMNS

    low_flow_dataset_name = "20250402_20X"
    high_flow_dataset_name = "20250611_20X"
    dataset_name_list = [low_flow_dataset_name, high_flow_dataset_name]

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

        df = load_pc_diffae_liveseg_feats_merged_table(dataset_name)
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

        # test_timepoints = sorted(df_subset.image_index.unique())[:5]
        # df_subset = df_subset[df_subset["image_index"] == test_timepoints]

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

    # def initial_test():
    #     x_slice = slice(record.start_x_cdh5_seg.values.item(), record.end_x_cdh5_seg.values.item())
    #     y_slice = slice(record.start_y_cdh5_seg.values.item(), record.end_y_cdh5_seg.values.item())

    #     # plt.imshow(overlay[y_slice, x_slice])
    #     # plt.show()
    #     # plt.clf()

    #     overlay2 = label2rgb(
    #         find_boundaries(seg_arr == seg_of_interest),
    #         rescale_intensity(np.clip(raw_arr, a_min=20, a_max=150), out_range=(0, 1)),
    #         bg_label=0,
    #         alpha=0.3,
    #     )
    #     plt.imshow(overlay2[y_slice, x_slice])
    #     plt.show()
    #     plt.clf()

    #     seg_bound = find_boundaries(seg_arr == seg_of_interest)
    #     seg_bound_locs = np.where(seg_bound)
    #     crop = tuple(slice(locs_dim.min(), locs_dim.max() + 1) for locs_dim in seg_bound_locs)
    #     plt.imshow(seg_bound[crop])
    #     plt.show()
    #     plt.clf()

    #     seg_centroid = tuple(int(locs_dim.mean()) for locs_dim in seg_bound_locs)

    #     # get the angle from each pixel in seg_bound to seg_centroid and also
    #     # the fluorescence intensity at each of those pixels
    #     angles = np.arctan2(
    #         seg_bound_locs[0] - seg_centroid[0],
    #         seg_bound_locs[1] - seg_centroid[1],
    #     )
    #     intensities = raw_arr[seg_bound_locs]

    #     plt.scatter(angles, intensities, color="k")
    #     plt.show()
    #     plt.clf()

    #     import seaborn as sns

    #     # Are the distributions of intensities different between high and low AT ALL?
    #     sns.histplot(intensities)

    #     intensities.std()

    #     # take peak of histogram (use mean instead?) to get baseline of fluorescence intensity
    #     # compare with 95th percentile of intensities(??) to determine if there is a "puncta"
    #     # of high fluorescence at the cell boundary(??)

    #     # make sure to filter to steady state only
    #     # pick just 1 position to start with for low and high each


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
