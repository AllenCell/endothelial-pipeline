from enum import StrEnum


class ColumnName:
    """Dataframe column names. Base level column names are shared among dataframes."""

    DATASET = "dataset"
    """Name of dataset."""

    POSITION = "position"
    """Zarr file position (FOV) of the corresponding segmentation or patch (crop)."""

    TIMEPOINT = "frame_number"
    """Timepoint (unit = frames)."""

    TRACK_ID = "track_id"
    """Track ID assigned by the tracking algorithm."""

    CROP_INDEX = "crop_index"
    """Unique integer index of a patch (crop) trajectory across dataset, FOV, and XY-position."""

    TRACK_LENGTH = "track_duration"
    """Number of timepoints in the track."""

    ZARR_PATH = "zarr_path"
    """Path to Zarr file for image that patch (crop) comes from."""

    IMAGE_SIZE_X = "image_size_x"
    """Size of the whole image in the X dimension in pixels."""

    IMAGE_SIZE_Y = "image_size_y"
    """Size of the whole image in the Y dimension in pixels."""

    PIXEL_SIZE_XY_IN_UM = "pixel_size_xy_in_um"
    """Size of a pixel in the XY dimensions in microns."""

    TIME_RESOLUTION_MINUTES = "time_resolution_minutes"
    """Time resolution of the timelapse in minutes."""

    DURATION = "duration_minutes"
    """Timelapse duration in timeframes"""

    SHEAR_STRESS = "shear_stress"
    """Shear stress value in dyn/cm^2."""

    SHEAR_STRESS_REGIME = "shear_stress_regime"
    """Experimentalist categorization of the shear stress response."""

    CDH5_CHANNEL_INDEX_ZARR = "cdh5_channel_index_in_zarr"
    """Index of the CDH5 channel in the Zarr file."""

    BF_CHANNEL_INDEX_ZARR = "brightfield_channel_index_zarr"
    """Index of the brightfield channel in the Zarr file."""

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

        NEMATIC_ORDER = "nematic_order"
        """Column name for nematic order (computed as `cos(2*theta)`)."""

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
        """The total number of nuclei present at the timepoint of interest."""

        # DiffAE and crop-based feature columns
        NUM_NUCLEI_IN_CROP = "num_nuclei_in_crop"
        """Number of label-free predicted nuclei in a crop at a particular timepoint."""

        LABELS_IN_CROP = "all_labels_in_crop"
        """List of all cell segmentation labels present in a crop at a particular timepoint."""

        START_X_RES_0 = "start_x_resolution_0"
        """x-coordinate defining the beginning of the crop at resolution level 0 (the native resolution)."""

        END_X_RES_0 = "end_x_resolution_0"
        """x-coordinate defining the end of the crop at resolution level 0 (the native resolution)."""

        START_Y_RES_0 = "start_y_resolution_0"
        """y-coordinate defining the beginning of the crop at resolution level 0 (the native resolution)."""

        END_Y_RES_0 = "end_y_resolution_0"
        """y-coordinate defining the end of the crop at resolution level 0 (the native resolution)."""

        CROP_SIZE = "crop_size"
        """Size of the crop in pixels at resolution level 0 (the native resolution)."""

        RESOLUTION_FOR_DIFFAE = "diffae_resolution_level_to_use"
        """Resolution level to use for DiffAE features.
        This is used to downsample the resolution 0 x and y coordinates and crop sizes
        when constructing the dataframe used to extract DiffAE features from track-based
        crops."""

        # temporal features: Time-related column names.
        TIME_HRS = "time_hours"
        """The time in hours since the start of the timelapse."""

        TIME_MINS = "time_minutes"
        """The time in minutes since the start of the timelapse."""

        NORMALIZED_TIME_PER_TRACK = "normalized_time"
        """The time normalized to the track duration (from 0 to 1)."""

        TIME_HRS_SINCE_FLOW = "time_hours_since_flow_start"
        """The time in hours since cells first start experiencing a shear stress."""

        # morphological features
        ORIENTATION = "orientation"
        """Orientation of the cell in radians ranging from 0 to π, where 0
        corresponds to the cell being oriented along the positive x-axis
        and π means the cell is oriented along the negative x-axis.
        """

        ORIENTATION_DEG = "orientation_deg"
        """Orientation of the cell in degrees ranging from 0 to 180, where 0
        corresponds to the cell being oriented along the positive x-axis
        and 180 means the cell is oriented along the negative x-axis.
        """

        ALIGNMENT = "alignment"
        """Alignment of the cell orientation relative to the flow direction,
        where 0 means the cell is aligned parallel to flow and 90 means the cell
        is aligned perpendicular to flow."""

        ALIGNMENT_DEG = "alignment_deg"
        """Alignment of the cell orientation relative to the flow direction in degrees,
        where 0 means the cell is aligned parallel to flow and 90 means the cell
        is aligned perpendicular to flow."""

        NEMATIC_ORDER = "nematic_order"
        ECCENTRICITY = "eccentricity"
        ASPECT_RATIO = "aspect_ratio"
        MAJOR_AXIS = "major_axis_length"
        """The length of the major axis of an ellipse fitted to the cell segmentation."""

        MINOR_AXIS = "minor_axis_length"
        """The length of the minor axis of an ellipse fitted to the cell segmentation."""

        SOLIDITY = "solidity"
        """Solidity of the cell, defined as the ratio of the cell area to the area of its convex hull."""

        AREA_UM_SQ = "area_um_squared"
        """Area of the cell segmentation in microns squared (um²)."""

        PERIMETER_UM = "perimeter_um"
        """Perimeter of the cell segmentation in microns."""

        AREA_PX_SQ = "area_px_squared"
        """Area of the cell segmentation in pixels squared."""

        PERIMETER_PX = "perimeter_px"
        """Perimeter of the cell segmentation in pixels."""

        NUCLEI_POSITION_X = "nuclei_position_x"
        NUCLEI_POSITION_Y = "nuclei_position_y"
        NUCLEI_POSITION_X_UM = "nuclei_position_x_um"
        NUCLEI_POSITION_Y_UM = "nuclei_position_y_um"
        NUCLEI_POSITION_ANGLE = "nuclei_position_angle"
        NUCLEI_POSITION_ANGLE_DEG = "nuclei_position_angle_deg"
        NUCLEI_POSITION_DISTANCE = "nuclei_position_distance"
        NUCLEI_LABEL = "nuclei_seg_with_most_overlap_0"
        NUCLEI_CENTROID_X = "nuc_with_most_overlap_0_centroid_X"
        NUCLEI_CENTROID_Y = "nuc_with_most_overlap_0_centroid_Y"

        # fluorescence features
        EDGE_FLUOR = "edge_fluorescence_au"
        """List of the fluorescence values along the boundary between 2 cell
        segmentations in arbitrary units (au) (excluding pixels at nodes)."""

        NODE_FLUOR = "node_fluorescence_au"
        """List of the fluorescence values at the junctions between 3 or more
        cell segmentations (i.e. nodes) in arbitrary units (au)."""

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
        CENTROID_X_UM = "centroid_x_um"
        CENTROID_Y_UM = "centroid_y_um"

        # dynamic features (values depend on how dataframes are filtered)
        CENTROID_VELOCITY_X_UM_PER_MIN = "centroid_velocity_x_um_per_min"
        CENTROID_VELOCITY_Y_UM_PER_MIN = "centroid_velocity_y_um_per_min"
        CENTROID_VELOCITY_ANGLE = "centroid_velocity_angle"
        CENTROID_VELOCITY_ANGLE_DEG = "centroid_velocity_angle_deg"
        CENTROID_VELOCITY_UM_PER_MIN = "centroid_velocity_um_per_min"
        ALIGNMENT_VELOCITY_DEG = "alignment_angular_velocity_deg"
        NUCLEI_POSITION_RELATIVE_MIGRATION_DEG = "nuclei_position_relative_migration_angle_deg"
        NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD = "nuclei_position_vs_migration_angle_dotproduct"
        CHANGE_IN_NUM_NUCLEI_IN_CROP_PER_MIN = "dnum_nuclei_in_crop_dt_mins"
        VELOCITY_ANGLES_IN_CROP = "all_velocity_angles_in_crop_deg"
        VELOCITY_UM_PER_MIN_IN_CROP = "all_velocity_um_per_min_in_crop"
        VECTOR_MEAN_FOR_CROP_ANGLE = "vector_mean_for_crop_angle"
        VECTOR_MEAN_FOR_CROP_MAGNITUDE = "vector_mean_for_crop_magnitude"
        CHANGE_IN_FLUOR_PER_MIN_CELL = "dmean_fluor_intensity_dt_cell"
        CHANGE_IN_FLUOR_PER_MIN_EDGE = "dmean_fluor_intensity_dt_edge"
        CHANGE_IN_FLUOR_PER_MIN_NODE = "dmean_fluor_intensity_dt_node"

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
        UNFED = "unfed"
        XY_SHIFT = "xy_shift"
        Z_SHIFT = "z_shift"

        BF_MEAN_INTENSITY = "bf_mean_intensity"
        """Flattened mean BF intensity over XY."""

        BF_ROLLING_MEDIAN = "bf_rolling_median"
        """Rolling median over BF mean intensity."""

        BF_DARK_THRESHOLD = "bf_dark_threshold"
        """BF intensity dark outlier threshold."""

        BF_PARTIAL_DARK_THRESHOLD = "bf_partial_dark_threshold"
        """BF intensity partial dark outlier threshold."""

        BF_BRIGHT_THRESHOLD = "bf_bright_threshold"
        """BF intensity bright outlier threshold."""

        BF_DARK_OUTLIERS = "bf_dark_outliers"
        """ndices of BF intensity dark outliers."""

        BF_PARTIAL_DARK_OUTLIERS = "bf_partial_dark_outliers"
        """ndices of BF intensity partial dark outliers."""

        BF_BRIGHT_OUTLIERS = "bf_bright_outliers"
        """Indices of BF intensity bright outliers."""

        GFP_TIMEPOINT_MEANS = "gfp_timepoint_means"
        """Mean GFP intensity intensity over XY per timepoint."""

        GFP_ROLLING_MEDIAN = "gfp_rolling_median"
        """Rolling median over GFP mean intensity."""

        GFP_LOWER_THRESHOLD = "gfp_lower_threshold"
        """GFP intensity lower threshold."""

        GFP_UPPER_THRESHOLD = "gfp_upper_threshold"
        """GFP intensity upper threshold."""

        GFP_DARK_OUTLIERS = "gfp_dark_outliers"
        """Indices of GFP intensity dark outliers."""

        GFP_BRIGHT_OUTLIERS = "gfp_bright_outliers"
        """Indices of GFP intensity bright outliers."""

        CENTER_PLANES = "center_planes"
        """List of center planes determined by minimum standard deviation."""

        CENTER_PLANE_MEAN = "center_plane_mean"
        """Mean of center plane across timepoints."""

        CENTER_PLANE_STD_DEV = "center_plane_std_dev"
        """Standard deviation of center plane across timepoints."""

        CENTER_PLANE_SLICES_STD_DEVS = "center_plane_slice_std_devs"
        """List of slices standard devisions for representative timepoint."""

    class OpticalFlow(StrEnum):
        """Dataframe column names used in the optical-flow feature workflow."""

        # --- Base feature names (before dt suffix) produced by compute_flow_statistics ---
        SPEED_MEAN_BASE = "optical_flow_mean_speed"
        """Base name for mean speed (before dt suffix)."""

        UNIT_VECTOR_MEAN_BASE = "optical_flow_mean_unit_vector"
        """Base name for mean unit vector coherence (before dt suffix)."""

        SPEED_STD_BASE = "optical_flow_std_speed"
        """Base name for speed standard deviation (before dt suffix)."""

        ANGLE_MEAN_BASE = "optical_flow_mean_angle"
        """Base name for mean angle (before dt suffix)."""

        ANGLE_STD_BASE = "optical_flow_angle_std"
        """Base name for angle standard deviation (before dt suffix)."""

        U_MEAN_BASE = "optical_flow_mean_u"
        """Base name for mean u component (before dt suffix)."""

        V_MEAN_BASE = "optical_flow_mean_v"
        """Base name for mean v component (before dt suffix)."""

        U_STD_BASE = "optical_flow_std_u"
        """Base name for u standard deviation (before dt suffix)."""

        V_STD_BASE = "optical_flow_std_v"
        """Base name for v standard deviation (before dt suffix)."""

        SPEED_ABOVE_1_COUNT_BASE = "speed_above_1_count"
        """Base name for count of pixels above speed threshold (before dt suffix)."""

        UNIT_VECTOR_MEAN_FAST_BASE = "optical_flow_mean_unit_vector_fast"
        """Base name for fast-pixel unit vector coherence (before dt suffix)."""

        RADIAL_COHERENCE_BASE = "optical_flow_radial_coherence"
        """Base name for radial coherence (before dt suffix)."""

        RADIAL_COHERENCE_WEIGHTED_BASE = "optical_flow_radial_coherence_weighted"
        """Base name for distance-weighted radial coherence (before dt suffix)."""

        # --- Final (suffixed) feature names used in dataframes ---
        SPEED_MEAN = "optical_flow_mean_speed_dt1"
        """Mean speed of the optical flow vectors in a crop."""

        UNIT_VECTOR_MEAN = "ema01_optical_flow_mean_unit_vector_dt1"
        """Mean unit vector of the optical flow vectors in a crop. EMA smoothing with alpha=0.01."""

        SPEED_STD = "optical_flow_std_speed_dt1"
        """Standard deviation of the speeds of the optical flow vectors in a crop."""

        ANGLE_MEAN = "optical_flow_mean_angle_dt1"
        """Mean angle of the optical flow vectors in a crop."""

        ANGLE_STD = "optical_flow_angle_std_dt1"
        """Standard deviation of the angles of the optical flow vectors in a crop."""

        U_MEAN = "optical_flow_mean_u_dt1"
        """Mean u (x) component of the optical flow vectors in a crop."""

        V_MEAN = "optical_flow_mean_v_dt1"
        """Mean v (y) component of the optical flow vectors in a crop."""

        U_STD = "optical_flow_std_u_dt1"
        """Standard deviation of the u (x) components of the optical flow vectors in a crop."""

        V_STD = "optical_flow_std_v_dt1"
        """Standard deviation of the v (y) components of the optical flow vectors in a crop."""

        SPEED_ABOVE_1_COUNT = "speed_above_1_count_dt1"
        """Number of pixels whose speed exceeds the threshold (fast-coherence feature)."""

        UNIT_VECTOR_MEAN_FAST = "optical_flow_mean_unit_vector_fast_dt1"
        """Mean unit vector coherence computed only over fast pixels."""

        RADIAL_COHERENCE = "optical_flow_radial_coherence_dt1"
        """Mean dot product of unit flow with unit radial vector from crop centre."""

        RADIAL_COHERENCE_WEIGHTED = "optical_flow_radial_coherence_weighted_dt1"
        """Distance-weighted radial coherence."""

        # --- EMA-smoothed and unsuffixed variants used in plotting / TFE viewer ---
        UNIT_VECTOR_MEAN_RAW = "optical_flow_mean_unit_vector_dt1"
        """Mean unit vector coherence (no EMA smoothing)."""

        EMA005_UNIT_VECTOR_MEAN = "ema005_optical_flow_mean_unit_vector_dt1"
        """Mean unit vector coherence with EMA smoothing, alpha=0.05."""

        EMA02_UNIT_VECTOR_MEAN = "ema02_optical_flow_mean_unit_vector_dt1"
        """Mean unit vector coherence with EMA smoothing, alpha=0.2."""

        EMA005_UNIT_VECTOR_MEAN_FAST = "ema005_optical_flow_mean_unit_vector_fast_dt1"
        """Mean unit vector coherence over fast pixels with EMA smoothing, alpha=0.05."""

        EMA01_UNIT_VECTOR_MEAN_FAST = "ema01_optical_flow_mean_unit_vector_fast_dt1"
        """Mean unit vector coherence over fast pixels with EMA smoothing, alpha=0.1."""

        EMA02_UNIT_VECTOR_MEAN_FAST = "ema02_optical_flow_mean_unit_vector_fast_dt1"
        """Mean unit vector coherence over fast pixels with EMA smoothing, alpha=0.2."""

        EMA01_RADIAL_COHERENCE = "ema01_optical_flow_radial_coherence_dt1"
        """Radial coherence with EMA smoothing, alpha=0.1."""

        EMA01_RADIAL_COHERENCE_WEIGHTED = "ema01_optical_flow_radial_coherence_weighted_dt1"
        """Distance-weighted radial coherence with EMA smoothing, alpha=0.1."""

    class VectorField(StrEnum):
        """Column name suffixes used in vector field / dynamics analysis."""

        FIXED_POINT_INDEX = "fixed_point_id"
        """Column name for the index of the fixed point in the fixed point dataframe."""

        STABILITY = "stability"
        """Stability classification of a fixed point."""

        FIXED_POINT_PREFIX = "fp_"
        """Prefix for column names representing coordinates of fixed points in feature space."""

        DRIFT = "drift"
        """Column name denoting the drift in a given variable."""

        DISTANCE_FROM_FP_PREFIX = "dist_from_fp_"
        """Prefix for column names representing the distance from a fixed point in N-D space."""

        DISTANCE_FROM_FP_1D_SIGNED_PREFIX = "diff_from_fp_"
        """Prefix for column names representing the signed difference from a fixed point along a single dimension."""

        FPT_DISTANCE_THRESHOLD = "fpt_distance_threshold"
        """Column name for the distance threshold used to determine whether a trajectory has reached a fixed point in first passage time analysis."""

        IS_AT_FP_PREFIX = "is_at_fp_"
        """Prefix for column names indicating whether a data point is at a fixed point."""

        TRAJ_REACHED_FP_PREFIX = "traj_reached_fp_"
        """Prefix for column names indicating whether a trajectory reached a fixed point."""

        FIRST_PASSAGE_DIST_PREFIX = "first_passage_dist_from_fp_"
        """Prefix for column names representing the distance from a fixed point at which a
        trajectory first passed the threshold for being considered to have reached the fixed point."""

        FIRST_PASSAGE_TIME_SUFFIX = "_first_passage_time"
        """Suffix for column names representing the first passage time to a fixed point."""

        FIRST_PASSAGE_PREFIX = "first_passage_"
        """Prefix for column names representing the first passage distance or time to a fixed point."""

        TIME_TO_FP_PREFIX = "time_to_fp_"
        """Prefix for column names representing the time to until a fixed point is reached."""

        BIN_SIZE_PREFIX = "bin_size_"
        """Column name for the sizes of bins used when discretizing feature space."""

        BIN_LIMITS_PREFIX = "bin_limits_"
        """Column name for the limits of bins used when discretizing feature space."""

        BIN_CENTER = "bin_center"
        """Column name for the center of bins used when discretizing feature space."""

        BIN_EDGES = "bin_edges"
        """Column name for the edges of bins used when discretizing feature space."""

        BIN_INDEX = "bin_index"
        """Column name for the index of the bin that a data point falls into when feature space is discretized."""

        FPT_METRIC = "fpt_metric"
        """Column name for the metric used in the first passage time analysis."""

        PERCENT_TRAJ_APPROACHED_FP = "percent_trajectories_approached_fp"
        """Column name for the percentage of trajectories that approached a fixed point within a certain distance threshold."""

        LINEFIT_INTERCEPT_ODR = "intercept_odr"
        """Column name for the intercept of a line fit to the relationship between first passage time and distance from the fixed point using orthogonal distance regression (ODR)."""

        LINEFIT_REDUCED_CHI_SQUARED_ODR = "reduced_chi_squared_odr"
        """Column name for the reduced chi-squared value of a line fit to the relationship between first passage time and distance from the fixed point using orthogonal distance regression (ODR)."""

        PEARSON_R = "r_value_pearson"
        """Column name for the Pearson correlation coefficient between first passage time and distance from the fixed point."""

        LINEFIT_SLOPE = "slope_odr"
        """Column name for the slope of a line fit to the relationship between first passage time to the fixed point for grid and tracked crops."""

    class BootstrapAnalysis(StrEnum):
        """Column name suffixes used in bootstrap fixed-point analysis."""

        DETECTION_RATE = "detection_rate"
        """Fraction of bootstrap iterations in which a matched fixed point was found."""

        CLUSTER_MEAN = "cluster_mean"
        """Mean coordinate of matched bootstrap fixed points."""

        CI_LOWER = "ci_lower"
        """Lower bound of the bootstrap confidence interval."""

        CI_UPPER = "ci_upper"
        """Upper bound of the bootstrap confidence interval."""

    class AutoCorrelation(StrEnum):
        """Column name suffixes used in autocorrelation analysis."""

        LAG = "lag"
        """Column name for the lag at which the autocorrelation is computed."""

        FEATURE = "feature"
        """Column name indicating the feature variable for which autocorrelation is computed."""

        ACF_MEAN = "autocorrelation_mean"
        """Column name for the mean autocorrelation value across tracks at a given lag."""

        ACF_LOWER_PERCENTILE = "autocorrelation_lower_percentile"
        """Column name for the lower percentile of autocorrelation values across tracks at a given lag."""

        ACF_UPPER_PERCENTILE = "autocorrelation_upper_percentile"
        """Column name for the upper percentile of autocorrelation values across tracks at a given lag."""

        EXPONENTIAL_FIT = "exponential_fit"
        """Column name for the values of the evaluated exponential fit curve at a given lag."""

    class ModelQC(StrEnum):
        """Dataframe column names used in the Model-QC metrics parquet."""

        RANDOM_SEED = "random_seed"
        """Column name for the noise/RNG seed used for a given inference row."""

        EXAMPLE_SET = "example_set"
        """Column name for the curated example-set label (e.g. ``rep_2_positions``)."""

        EXAMPLE_IDX = "example_idx"
        """Column name for the 0-based position of the example within its set."""

        DATASET_NAME = "dataset_name"
        """Column name for the source dataset name of the example image."""


ColumnNameType = (
    str
    | ColumnName
    | ColumnName.DiffAEData
    | ColumnName.SegData
    | ColumnName.SegDataFilters
    | ColumnName.SegDataWorkflowVerification
    | ColumnName.Annotations
    | ColumnName.OpticalFlow
    | ColumnName.BootstrapAnalysis
    | ColumnName.VectorField
    | ColumnName.AutoCorrelation
    | ColumnName.ModelQC
)
"""Type hint for all column name enums."""
