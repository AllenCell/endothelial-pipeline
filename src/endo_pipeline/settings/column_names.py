from enum import StrEnum


class ColumnName:
    """Dataframe column names. Base level column names are shared among dataframes."""

    DATASET = "dataset"
    """Column name for dataset name."""

    POSITION = "position"
    """Column name for position identifier."""

    TIMEPOINT = "frame_number"
    """Column name for timepoint (frame number)."""

    TRACK_ID = "track_id"
    """The track ID assigned by the tracking algorithm."""

    CROP_INDEX = "crop_index"
    """Column name for crop index."""

    TRACK_LENGTH = "track_duration"
    """Column name for track length (number of timepoints in the track)."""

    ZARR_PATH = "zarr_path"
    """Column name for path to the source Zarr file."""

    IMAGE_SIZE_X = "image_size_x"
    """The size of the image in the X dimension (pixels)."""

    IMAGE_SIZE_Y = "image_size_y"
    """The size of the image in the Y dimension (pixels)."""

    PIXEL_SIZE_XY_IN_UM = "pixel_size_xy_in_um"
    """The size of a pixel in the XY dimensions in microns."""

    TIME_RESOLUTION_MINUTES = "time_resolution_minutes"
    """The time resolution of the timelapse in minutes."""

    DURATION = "duration"
    """Timelapse duration in timeframes"""

    SHEAR_STRESS = "shear_stress"
    """The shear stress value in dyn/cm^2."""

    SHEAR_STRESS_REGIME = "shear_stress_regime"
    """Experimentalist categorization of the shear stress response (e.g. "low flow")."""

    CDH5_CHANNEL_INDEX_ZARR = "cdh5_channel_index_in_zarr"

    BF_CHANNEL_INDEX_ZARR = "brightfield_channel_index_zarr"

    class DiffAEData(StrEnum):
        """Dataframe column names used in DiFFAE feature dataframes."""

        LATENT_FEATURE_PREFIX = "feat_"
        """Prefix for latent feature column names."""

        PCA_FEATURE_PREFIX = "pc_"
        """Prefix for PCA-transformed feature column names."""

        POLAR_RADIUS = "polar_r"
        """Column name for polar radius coordinate computed from PC1 and PC2."""

        POLAR_ANGLE = "polar_theta"
        """Column name for polar angle coordinate computed from PC1 and PC2."""

        PC3_FLIPPED = "rho"
        """Column name for PC3 value with sign flipped as proxy measure of cell density."""

        DIFFERENCE_SUFFIX = "_diff"
        """Suffix for columns representing differences between feature values."""

        MODEL_MANIFEST = "model_manifest_name"
        """Column name for model manifest name."""

        MODEL_RUN = "run_name"
        """Column name for model run name."""

        RESOLUTION = "resolution_level"
        """Column name for resolution level of the image."""

        START_X = "start_x"
        """Upper-left x-coordinate of the crop."""

        START_Y = "start_y"
        """Upper-left y-coordinate of the crop."""

        END_X = "end_x"
        """Lower-right x-coordinate of the crop."""

        END_Y = "end_y"
        """Lower-right y-coordinate of the crop."""

        CROP_SIZE_X = "crop_size_x"
        """Width of the crop in pixels."""

        CROP_SIZE_Y = "crop_size_y"
        """Height of the crop in pixels."""

    class SegData(StrEnum):
        """Dataframe column names used in segmentation-based feature dataframes."""

        # dataset and segmentation information columns
        LABEL = "label"
        """The cell segmentation ID.
        Note that this is different from the track ID, and can change from one timepoint to the next.
        """

        NUM_TRACKS_AFTER_FILTERING = "num_unique_tracks_after_filtering_at_T"
        """The number of unique tracks that pass filtering criteria at the timepoint of interest."""

        NUM_TRACKS_BEFORE_FILTERING = "num_unique_tracks_before_filtering_at_T"
        """The number of unique tracks that are present at the timepoint of interest before applying
        filtering criteria. The same as the number of segmentations at that timepoint.
        """

        NUM_NUCLEI_AT_TIMEPOINT = "total_nuclei_count_at_T"

        # DiffAE and crop-based feature columns
        NUM_NUCLEI_IN_CROP = "num_nuclei_in_crop"
        LABELS_IN_CROP = "all_labels_in_crop"
        START_X = "start_x_resolution_0"
        END_X = "end_x_resolution_0"
        START_Y = "start_y_resolution_0"
        END_Y = "end_y_resolution_0"
        CROP_SIZE = "crop_size"
        RESOLUTION_FOR_DIFFAE = "diffae_resolution_level_to_use"

        # temporal features: Time-related column names.
        TIME_HRS = "time_hours"
        TIME_MINS = "time_minutes"
        TIME_HRS_SINCE_FLOW = "time_hours_since_flow_start"

        # morphological features
        ORIENTATION = "orientation"
        ORIENTATION_DEG = "orientation_deg"
        ALIGNMENT = "alignment"
        ALIGNMENT_DEG = "alignment_deg"
        NEMATIC_ORDER = "nematic_order"
        ECCENTRICITY = "eccentricity"
        ASPECT_RATIO = "aspect_ratio"
        MAJOR_AXIS = "major_axis_length"
        MINOR_AXIS = "minor_axis_length"
        SOLIDITY = "solidity"
        AREA = "area_um_squared"
        PERIMETER = "perimeter_um"
        AREA_PX_SQ = "area_px_squared"
        PERIMETER_PX = "perimeter_px"
        NUCLEI_POSITION_X = "nuclei_position_x"
        NUCLEI_POSITION_Y = "nuclei_position_y"
        NUCLEI_POSITION_ANGLE = "nuclei_position_angle"
        NUCLEI_POSITION_ANGLE_DEG = "nuclei_position_angle_deg"
        NUCLEI_POSITION_DISTANCE = "nuclei_position_distance"
        NUCLEI_LABEL = "nuclei_seg_with_most_overlap_0"
        NUCLEI_CENTROID_X = "nuc_with_most_overlap_0_centroid_X"
        NUCLEI_CENTROID_Y = "nuc_with_most_overlap_0_centroid_Y"

        # fluorescence features
        EDGE_FLUOR = "edge_fluorescence_au"
        NODE_FLUOR = "node_fluorescence_au"

        CELL_FLUOR_MEAN = "cell_fluorescence_mean_au"
        CELL_FLUOR_STD = "cell_fluorescence_std_au"
        CELL_FLUOR_MEDIAN = "cell_fluorescence_median_au"
        CELL_FLUOR_MIN = "cell_fluorescence_min_au"
        CELL_FLUOR_MAX = "cell_fluorescence_max_au"
        CELL_FLUOR_PCT25 = "cell_fluorescence_pct25_au"
        CELL_FLUOR_PCT75 = "cell_fluorescence_pct75_au"

        EDGE_FLUOR_MEAN = "edge_fluorescence_mean_au"
        EDGE_FLUOR_STD = "edge_fluorescence_std_au"

        NODE_FLUOR_MEAN = "node_fluorescence_mean_au"
        NODE_FLUOR_STD = "node_fluorescence_std_au"

        EDGE_AND_NODE_FLUOR_MEAN = "edge_and_node_fluorescence_mean_au"
        EDGE_AND_NODE_FLUOR_STD = "edge_and_node_fluorescence_std_au"

        # other features:
        NUM_NEIGHBORS = "number_of_neighbors"
        NEIGHBOR_LABELS = "neighboring_cell_labels"
        CENTROID = "centroid"
        CENTROID_X = "centroid_x"
        CENTROID_Y = "centroid_y"

    class SegDataFilters(StrEnum):
        """Filter-related column names for segmentation-based features."""

        IS_INCLUDED = "is_included"
        """Whether or not a track passes all filtering criteria and is included in the final dataset"""

        IS_EDGE_SEGMENTATION = "is_edge_segmentation"
        """Whether or not a segmentation touches the edge of the image"""

        IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE = "is_less_than_max_smoothed_area_normd_change"
        SMOOTHED_AREA_NORMD_DIFF = "smoothed_area_normd_diff"
        MAX_SMOOTHED_AREA_NORMALIZED_CHANGE = "max_smoothed_area_normd_change"

        IS_GREATER_THAN_MIN_TRACK_DURATION = "is_greater_than_min_track_duration"
        MIN_TRACK_DURATION = "min_track_duration"

        HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK = (
            "has_more_than_min_num_valid_points_per_track"
        )
        MIN_NUM_VALID_TIMEPOINTS_PER_TRACK = "min_num_valid_tp_per_track"
        NUM_VALID_TIMEPOINTS_IN_TRACK = "num_valid_tp_per_track"

        IS_VALID_BBOX = "bbox_is_in_bounds"

    class SegDataWorkflowVerification(StrEnum):
        """Column names for workflow development and verification checks."""

        SEGMENTATION_PATH = "filepath_segmentation_image"
        TRACKING_REF_IDX = "reference_index"
        TRACKING_MATCHED_QUERY_LABEL = "matched_query_label"
        TRACKING_OPTIMIZED_METRIC_VAL = "optimized_metric_value"
        TRACKING_MATCHING_METHOD = "matching_method"
        NUM_NUC_WITH_MOST_OVERLAP = "num_nuclei_with_most_overlap"
        SMOOTHED_AREA_NORMALIZED = "smoothed_area_normd"
        SIGMA_FOR_AREA_SMOOTHING = "gaussian_sigma_for_area_smoothing"
        NUM_UNIQUE_TRACKS_PER_TIMEPOINT = "num_unique_tracks_per_timeframe"
        NODE_LABELS = "node_labels"
        EDGE_LABELS = "edge_labels"
        NODE_PAIR_LABELS = "node_pair_labels"
        NUCLEI_LABELS_IN_CDH5_SEGMENTATION = "nuclei_segmentation_labels"
        NUCLEI_FRACTION_IN_CDH5_SEGMENTATION = "nuclei_seg_in_cdh5_seg_frac"
        NUCLEI_INTENSITY_COLUMN_PREFIX = "nuc_seg_intens_"
        NUCLEI_SEG_LABEL_PREFIX = "nuclei_seg_with_most_overlap_"

    class Annotations(StrEnum):
        """Column names for manual annotations of segmentation quality and other features."""

        AUTO_BF_SCOPE_ERROR = "auto_bf_scope_error"
        AUTO_BF_TEMP_ARTIFACT = "auto_bf_temp_artifact"
        AUTO_GFP_SCOPE_ERROR = "auto_gfp_scope_error"
        BF_SCOPE_ERROR = "bf_scope_error"
        BF_TEMP_ARTIFACT = "bf_temp_artifact"
        GFP_SCOPE_ERROR = "gfp_scope_error"
        CELL_PILING = "cell_piling"
        NOT_STEADY_STATE = "not_steady_state"


class CytoDLLoadDataKeys(StrEnum):
    """Metadata keys passed into MultiDimImageDataset for loading images."""

    FILE_PATH = "path"
    """Key for path to the source image file."""

    TIMEPOINT = "T"
    """Key for timepoint (frame number)."""

    TIME_START = "frame_start"
    """Key for starting timepoint (frame number)."""

    TIME_END = "frame_stop"
    """Key for ending timepoint (frame number)."""

    TIME_STEP = "frame_step"
    """Key for timepoint (frame number) step size."""

    INCLUDE_TIMEPOINTS = "include_frames"
    """Key for list of timepoints (frame numbers) to include."""

    Z_START = "z_start"
    """Key for starting z-slice index."""

    Z_END = "z_stop"
    """Key for ending z-slice index."""

    Z_STEP = "z_step"
    """Key for z-slice index step size."""

    CHANNELS = "channel"
    """Key for list of channels to load."""

    RESOLUTION = "resolution_level"
    """Key for resolution level of the image."""

    SCENE = "scene"
    """Key for scene identifier."""

    START_X = ColumnName.DiffAEData.START_X
    """Upper-left x-coordinate of the crop."""

    START_Y = ColumnName.DiffAEData.START_Y
    """Upper-left y-coordinate of the crop."""

    END_X = ColumnName.DiffAEData.END_X
    """Lower-right x-coordinate of the crop."""

    END_Y = ColumnName.DiffAEData.END_Y
    """Lower-right y-coordinate of the crop."""


class CytoDLSaveDataKeys(StrEnum):
    """Metadata keys passed to CytoDL callback object SaveTabularData."""

    FILE_PATH = "filename_or_obj"
    """Key for path to the source image file."""

    TIMEPOINT = "T"
    """Key for timepoint (frame number)."""
