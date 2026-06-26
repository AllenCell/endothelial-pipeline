from enum import StrEnum


class ColumnName:
    """Dataframe column names. Base level column names are shared among dataframes."""

    DATASET = "dataset"
    """Name of dataset."""

    POSITION = "position"
    """Zarr file position (FOV) of the corresponding segmentation or patch."""

    TIMEPOINT = "frame_number"
    """Timepoint [frames]."""

    TRACK_ID = "track_id"
    """Track ID assigned by the tracking algorithm."""

    CROP_INDEX = "crop_index"
    """Unique integer index of a patch (crop) trajectory across dataset, FOV, and XY-position."""

    TRACK_LENGTH = "track_duration"
    """Number of timepoints in track."""

    ZARR_PATH = "zarr_path"
    """Path to Zarr file for image that the patch comes from."""

    IMAGE_SIZE_X = "image_size_x"
    """Size of the whole image in the X dimension [pixels]."""

    IMAGE_SIZE_Y = "image_size_y"
    """Size of the whole image in the Y dimension [pixels]."""

    PIXEL_SIZE_XY_IN_UM = "pixel_size_xy_in_um"
    """Size of a pixel in the XY dimensions [microns]."""

    TIME_RESOLUTION_MINUTES = "time_resolution_minutes"
    """Time resolution of the timelapse [minutes]."""

    DURATION = "duration_minutes"
    """Timelapse duration in timeframes"""

    SHEAR_STRESS = "shear_stress"
    """Shear stress value [dyn/cm^2]."""

    SHEAR_STRESS_REGIME = "shear_stress_regime"
    """Experimentalist categorization of the shear stress response."""

    CDH5_CHANNEL_INDEX_ZARR = "cdh5_channel_index_in_zarr"
    """Index of the CDH5 channel in the Zarr file."""

    BF_CHANNEL_INDEX_ZARR = "brightfield_channel_index_zarr"
    """Index of the brightfield channel in the Zarr file."""

    SOURCE_IMAGE_PATH_FOR_MODEL = "path"
    """Path to source image file for model training and evaluation."""

    IMAGE_CHANNELS_TO_LOAD_FOR_MODEL = "channel"
    """List of channels to load for model training and evaluation."""

    TIMEPOINTS_TO_INCLUDE_FOR_MODEL = "include_frames"
    """List of timepoints (frame numbers) to include for model training and evaluation."""

    Z_START_FOR_MODEL = "z_start"
    """Starting z-slice of image to use for model training and evaluation."""

    Z_END_FOR_MODEL = "z_stop"
    """Ending z-slice of image to use for model training and evaluation."""

    Z_STEP_FOR_MODEL = "z_step"
    """Step between Z-slices in image to use for model training and evaluation."""

    TIMEPOINT_FOR_MODEL = "T"
    """Timepoint (frame number) to use when loading images for model training and evaluation."""

    FRAME_START_FOR_MODEL = "frame_start"
    """First timepoint (frame number) to use when loading images for model training and evaluation."""

    FRAME_STOP_FOR_MODEL = "frame_stop"
    """Last timepoint (frame number) to use when loading images for model training and evaluation."""

    class DiffAEData(StrEnum):
        """Dataframe column names used in DiFFAE feature dataframes."""

        POLAR_RADIUS = "polar_r"
        """Transformed feature r (polar r from PC1 and PC2)."""

        POLAR_ANGLE = "polar_theta"
        """Transformed feature theta (rescaled polar angle from PC1 and PC2)."""

        NEMATIC_ORDER = "nematic_order"
        """Column name for nematic order (computed as `cos(2*theta)`)."""

        PC3_FLIPPED = "rho"
        """Transformed feature rho (negative PC3)."""

        DIFFERENCE_SUFFIX = "_diff"
        """Suffix for columns representing differences between feature values."""

        MODEL_MANIFEST = "model_manifest_name"
        """Manifest name of model used to generate latent vectors."""

        MODEL_RUN = "run_name"
        """Run name of model used to generate latent vectors."""

        RESOLUTION = "resolution_level"
        """Zarr resolution level used to generate patches for obtaining latent vectors."""

        START_X = "start_x"
        """X coordinate of upper left corner of patch [pixels] (zarr resolution level 1)."""

        START_Y = "start_y"
        """Y coordinate of upper left corner of patch [pixels] (zarr resolution level 1)."""

        END_X = "end_x"
        """X coordinate of lower right corner of patch [pixels] (zarr resolution level 1)."""

        END_Y = "end_y"
        """Y coordinate of lower right corner of patch [pixels] (zarr resolution level 1)."""

        CROP_SIZE_X = "crop_size_x"
        """Width of the patch [pixels] (zarr resolution level 1)."""

        CROP_SIZE_Y = "crop_size_y"
        """Height of the patch [pixels] (zarr resolution level 1)."""

    class SegData(StrEnum):
        """Dataframe column names used in segmentation-based feature dataframes."""

        LABEL = "label"
        """Cell segmentation ID (different from track ID, and may change between timepoints)."""

        NUM_TRACKS_AFTER_FILTERING = "num_unique_tracks_after_filtering_at_T"
        """Number of unique tracks that pass filtering criteria at timepoint of interest."""

        NUM_TRACKS_BEFORE_FILTERING = "num_unique_tracks_before_filtering_at_T"
        """Number of unique tracks present at timepoint of interest before filtering."""

        NUM_NUCLEI_AT_TIMEPOINT = "total_nuclei_count_at_T"
        """Total number of nuclei present at the timepoint of interest."""

        NUM_NUCLEI_IN_CROP = "num_nuclei_in_crop"
        """Number of label-free predicted nuclei in a crop/cell-centered patch."""

        LABELS_IN_CROP = "all_labels_in_crop"
        """List of all segmentation labels found within a crop/cell-centered patch."""

        START_X_RES_0 = "start_x_resolution_0"
        """X coordinate of upper left corner of patch [pixels] (zarr resolution level 0)."""

        END_X_RES_0 = "end_x_resolution_0"
        """X coordinate of lower right corner of patch [pixels] (zarr resolution level 0)."""

        START_Y_RES_0 = "start_y_resolution_0"
        """Y coordinate of upper left corner of patch [pixels] (zarr resolution level 0)."""

        END_Y_RES_0 = "end_y_resolution_0"
        """Y coordinate of lower right corner of patch [pixels] (zarr resolution level 0)."""

        CROP_SIZE = "crop_size"
        """Size of the crop [pixels] at resolution level 0 (the native resolution)."""

        RESOLUTION_FOR_DIFFAE = "diffae_resolution_level_to_use"
        """DiffAE feature resolution level used to downsample resolution 0 coordinates."""

        TIME_HRS = "time_hours"
        """Time since start of timelapse [hours]."""

        TIME_MINS = "time_minutes"
        """Time since start of timelapse [minutes]."""

        NORMALIZED_TIME_PER_TRACK = "normalized_time"
        """Time normalized to track duration (from 0 to 1)."""

        TIME_HRS_SINCE_FLOW = "time_hours_since_flow_start"
        """Time since cells first experienced a shear stress [hours]."""

        ORIENTATION = "orientation"
        """Orientation of cell [radians] from 0 (along +x axis) to pi (along -x axis)."""

        ORIENTATION_DEG = "orientation_deg"
        """Orientation of cell [degrees] from 0 (along +x axis) to 180 (along -x axis)."""

        ALIGNMENT = "alignment"
        """Alignment of cell relative to flow [radians] from 0 (parallel) to pi (perpendicular)."""

        ALIGNMENT_DEG = "alignment_deg"
        """Alignment of cell relative to flow [degrees] from 0 (parallel) to 90 (perpendicular)."""

        NEMATIC_ORDER = "nematic_order"
        """Nematic order of ellipse fit to cell segmentation from -1 (perpendicular) to 1 (parallel)."""

        ECCENTRICITY = "eccentricity"
        """Eccentricity of ellipse fit to cell segmentation (focal distance divided by major axis)."""

        ASPECT_RATIO = "aspect_ratio"
        """Aspect ratio of ellipse fit to cell segmentation (major axis divided by minor axis)."""

        MAJOR_AXIS = "major_axis_length"
        """Length of major axis of an ellipse fit to cell segmentation."""

        MINOR_AXIS = "minor_axis_length"
        """Length of minor axis of an ellipse fit to cell segmentation."""

        SOLIDITY = "solidity"
        """Solidity of cell (ratio of cell area to area of its convex hull)."""

        AREA_UM_SQ = "area_um_squared"
        """Area of cell segmentation [microns^2]."""

        PERIMETER_UM = "perimeter_um"
        """Perimeter of cell segmentation [microns]."""

        AREA_PX_SQ = "area_px_squared"
        """Area of cell segmentation [pixels^2]."""

        PERIMETER_PX = "perimeter_px"
        """Perimeter of cell segmentation [pixels]."""

        NUCLEI_POSITION_X = "nuclei_position_x"
        """X coordinate of the label-free nuclei prediction centroid [pixels]."""

        NUCLEI_POSITION_Y = "nuclei_position_y"
        """Y coordinate of the label-free nuclei prediction centroid [pixels]."""

        NUCLEI_POSITION_X_UM = "nuclei_position_x_um"
        """X coordinate of the label-free nuclei prediction centroid [microns]."""

        NUCLEI_POSITION_Y_UM = "nuclei_position_y_um"
        """Y coordinate of the label-free nuclei prediction centroid [microns]."""

        NUCLEI_POSITION_ANGLE = "nuclei_position_angle"
        """Angle of nucleus centroid from cell centroid relative to flow [radians] with 0 along +x and pi/-pi along -x."""

        NUCLEI_POSITION_ANGLE_DEG = "nuclei_position_angle_deg"
        """Angle of nucleus centroid from cell centroid relative to flow [degrees] with 0 along +x and 180/-180 along -x."""

        NUCLEI_POSITION_DISTANCE = "nuclei_position_distance"
        """Distance between nucleus centroid and cell segmentation centroid [pixels]."""

        NUCLEI_LABEL = "nuclei_with_most_overlap_0"
        """
        ID of nuclei prediction that overlaps most with cell segmentation (index
        0, multiple nuclei recorded in additional columns).
        """

        NUCLEI_CENTROID_X = "nuclei_with_most_overlap_0_centroid_X"
        """
        Centroid X coordinate of nuclei prediction that overlaps most with cell
        segmentation (index 0, multiple nuclei recorded in additional columns)
        """

        NUCLEI_CENTROID_Y = "nuclei_with_most_overlap_0_centroid_Y"
        """
        Centroid Y coordinate of nuclei prediction that overlaps most with cell
        segmentation (index 0, multiple nuclei recorded in additional columns)
        """

        EDGE_FLUOR = "edge_fluorescence_au"
        """Fluorescence [a.u.] along boundary between 2 cell segmentations (excluding pixels at nodes)."""

        NODE_FLUOR = "node_fluorescence_au"
        """Fluorescence [a.u.] at junctions between 3 or more cell segmentations."""

        CELL_FLUOR_MEAN = "cell_fluorescence_mean_au"
        """Mean of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        CELL_FLUOR_STD = "cell_fluorescence_std_au"
        """Standard deviation of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        CELL_FLUOR_MEDIAN = "cell_fluorescence_median_au"
        """Median of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        CELL_FLUOR_MIN = "cell_fluorescence_min_au"
        """Minimum of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        CELL_FLUOR_MAX = "cell_fluorescence_max_au"
        """Maximum of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        CELL_FLUOR_PCT25 = "cell_fluorescence_pct25_au"
        """25th percentile of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        CELL_FLUOR_PCT75 = "cell_fluorescence_pct75_au"
        """75th percentile of mEGFP-tagged VE-cadherin fluorescence [a.u.] in cytoplasmic region of cell segmentation."""

        EDGE_FLUOR_MEAN = "edge_fluorescence_mean_au"
        """Mean fluorescence [a.u.] at junctions between 2 cell segmentations (edge)."""

        EDGE_FLUOR_STD = "edge_fluorescence_std_au"
        """Standard deviation of fluorescence [a.u.] at junctions between 2 cell segmentations (edge)."""

        NODE_FLUOR_MEAN = "node_fluorescence_mean_au"
        """Mean fluorescence [a.u.] at junctions between 3 or more cell segmentations (node)."""

        NODE_FLUOR_STD = "node_fluorescence_std_au"
        """Standard deviation of fluorescence [a.u.] at junctions between 3 or more cell segmentations (node)."""

        EDGE_AND_NODE_FLUOR_MEAN = "edge_and_node_fluorescence_mean_au"
        """Mean fluorescence [a.u.] of cell segmentation edges and nodes."""

        EDGE_AND_NODE_FLUOR_STD = "edge_and_node_fluorescence_std_au"
        """Standard deviation of fluorescence [a.u.] of cell segmentation edges and nodes."""

        NUM_NEIGHBORS = "number_of_neighbors"
        """Number of unique cell segmentations that are adjacent to this one."""

        NEIGHBOR_LABELS = "neighboring_cell_labels"
        """Cell segmentation labels of adjacent cell segmentations."""

        CENTROID = "centroid"
        """Centroid of cell segmentation in (Y,X) [pixels]."""

        CENTROID_X = "centroid_x"
        """Centroid of cell segmentation X coordinate [pixels]."""

        CENTROID_Y = "centroid_y"
        """Centroid of cell segmentation Y coordinate [pixels]."""

        CENTROID_X_UM = "centroid_x_um"
        """Centroid of cell segmentation X coordinate [microns]."""

        CENTROID_Y_UM = "centroid_y_um"
        """Centroid of cell segmentation Y coordinate [microns]."""

        CENTROID_VELOCITY_X_UM_PER_MIN = "centroid_velocity_x_um_per_min"
        """Change in X coordinate of cell segmentation centroid [um/minute]."""

        CENTROID_VELOCITY_Y_UM_PER_MIN = "centroid_velocity_y_um_per_min"
        """Change in Y coordinate of cell segmentation centroid [um/minute]."""

        CENTROID_VELOCITY_ANGLE = "centroid_velocity_angle"
        """Angle of cell migration based on change in centroid [radians] with 0 along +x and pi/-pi along -x."""

        CENTROID_VELOCITY_ANGLE_DEG = "centroid_velocity_angle_deg"
        """Angle of cell migration based on change in centroid [degrees] with 0 along +x and 180/-180 along -x."""

        CENTROID_VELOCITY_UM_PER_MIN = "centroid_velocity_um_per_min"
        """Speed of cell migration based on change in cell segmentation centroid [um/minute]."""

        ALIGNMENT_VELOCITY_DEG = "alignment_angular_velocity_deg"
        """Rate of change of alignment angle [degrees/min]."""

        NUCLEI_POSITION_RELATIVE_MIGRATION_DEG = "nuclei_position_relative_migration_angle_deg"
        """
        Angle [degrees] of the nucleus centroid relative to cell segmentation
        centroid (counter-clockwise if positive, clockwise if negative, 0 if in
        the same direction as cell migration).
        """

        NUCLEI_POSITION_RELATIVE_MIGRATION_DOTPROD = "nuclei_position_vs_migration_angle_dotproduct"
        """Dot product of cell centroid-to-nuclei centroid vector and cell migration vector."""

        CHANGE_IN_NUM_NUCLEI_IN_CROP_PER_MIN = "dnum_nuclei_in_crop_dt_mins"
        """Change in number of label-free nuclei predictions found in a patch (crop) over time [number of nuclei/minute]."""

        VELOCITY_ANGLES_IN_CROP = "all_velocity_angles_in_crop_deg"
        VELOCITY_UM_PER_MIN_IN_CROP = "all_velocity_um_per_min_in_crop"
        VECTOR_MEAN_FOR_CROP_ANGLE = "vector_mean_for_crop_angle"
        VECTOR_MEAN_FOR_CROP_MAGNITUDE = "vector_mean_for_crop_magnitude"
        CHANGE_IN_FLUOR_PER_MIN_CELL = "dmean_fluor_intensity_dt_cell"
        """Change in mean fluorescence intensity of cytoplasmic region of cell segmentation over time [a.u./minute]."""

        CHANGE_IN_FLUOR_PER_MIN_EDGE = "dmean_fluor_intensity_dt_edge"
        """Change in mean fluorescence intensity of bicellular junction regions of cell segmentation over time [a.u./minute]."""

        CHANGE_IN_FLUOR_PER_MIN_NODE = "dmean_fluor_intensity_dt_node"
        """Change in mean fluorescence intensity of tricellular junction regions of cell segmentation over time [a.u./minute]."""

    class SegDataFilters(StrEnum):
        """Filter-related column names for segmentation-based features."""

        IS_INCLUDED = "is_included"
        """True if track passes all filtering criteria and is included in final dataset, False otherwise."""

        IS_EDGE_SEGMENTATION = "is_edge_segmentation"
        """True if segmentation touches edge of the image, False otherwise."""

        IS_LESS_THAN_MAX_SMOOTHED_AREA_NORMD_CHANGE = "is_less_than_max_smoothed_area_normd_change"
        """True if change in Gaussian-smoothed cell segmentation area is less than threshold, False otherwise."""

        SMOOTHED_AREA_NORMD_DIFF = "smoothed_area_normd_diff"
        """Change in Gaussian-smoothed cell segmentation area [pixels^2]."""

        MAX_SMOOTHED_AREA_NORMALIZED_CHANGE = "max_smoothed_area_normd_change"
        """Max change in normalized smoothed cell segmentation area before being considered invalid."""

        IS_GREATER_THAN_MIN_TRACK_DURATION = "is_greater_than_min_track_duration"
        """True if track duration exceeds minimum threshold, False otherwise."""

        MIN_TRACK_DURATION = "min_track_duration"
        """Minimum number of timepoints for a track to be kept."""

        HAS_MORE_THAN_MIN_NUM_VALID_POINTS_PER_TRACK = (
            "has_more_than_min_num_valid_points_per_track"
        )
        """True if track has sufficient number of timepoints passing all other filters, False otherwise."""

        MIN_NUM_VALID_TIMEPOINTS_PER_TRACK = "min_num_valid_tp_per_track"
        """Minimum number of timepoints that pass all other filters for a track to be kept."""

        NUM_VALID_TIMEPOINTS_IN_TRACK = "num_valid_tp_per_track"
        """Number of timepoints in track that pass all segmentation-based filters."""

        IS_VALID_BBOX = "bbox_is_in_bounds"
        """True if crop/cell-centered patch fits into the larger image without being clipped."""

    TRACKING_REFERENCE_INDEX = "reference_index"
    """Relative timepoint index used to compare nearby timepoints to find matching segmentation for track."""

    TRACKING_MATCHED_QUERY_LABEL = "matched_query_label"
    """List of matched labels from each timepoint starting from current one to up to the next 4 timepoints."""

    TRACKING_OPTIMIZED_METRIC_VALUE = "optimized_metric_value"
    """Value of metric (fraction of segmentation overlap between reference and query index timepoints) used for tracking."""

    TRACKING_MATCHING_METHOD = "matching_method"
    """Metric used by tracking algorithm to find matching segmentations."""

    class SegDataWorkflowVerification(StrEnum):
        """Column names for workflow development and verification checks."""

        NUM_NUC_WITH_MOST_OVERLAP = "num_nuclei_with_most_overlap"
        """Number of label-free predicted nuclei that overlap with a cell segmentation."""

        SMOOTHED_AREA_NORMALIZED = "smoothed_area_normd"
        """Area of cell segmentation after Gaussian smoothing across time [pixels^2]."""

        SIGMA_FOR_AREA_SMOOTHING = "gaussian_sigma_for_area_smoothing"
        """Standard deviation of Gaussian kernel used for smoothing cell segmentation areas across time."""

        NUM_UNIQUE_TRACKS_PER_TIMEPOINT = "num_unique_tracks_per_timeframe"
        """Number of unique cell segmentations per dataset per position per timepoint per track (should always be 1)."""

        NODE_LABELS = "node_labels"
        """Label IDs for each tricellular junction."""

        EDGE_LABELS = "edge_labels"
        """Label IDs of each edge of a segmentation boundary."""

        NODE_PAIR_LABELS = "node_pair_labels"
        """Pair of node label IDs that form each edge of a segmentation boundary."""

        CDH5_SEGMENTATION_LABEL = "cdh5_segmentation_label"
        """Label ID for the CDH5 cell segmentation."""

        NUCLEI_LABELS_IN_CDH5_SEGMENTATION = "nuclei_segmentation_labels"
        """List of labels from label-free nuclei predictions that overlap with the cell segmentation."""

        NUCLEI_FRACTION_IN_CDH5_SEGMENTATION = "nuclei_seg_in_cdh5_seg_frac"
        """List of fractions of label-free nuclei prediction that overlap with the cell segmentation."""

    class Annotations(StrEnum):
        """Column names for manual annotations of segmentation quality and other features."""

        AUTO_BF_SCOPE_ERROR = "auto_bf_scope_error"
        """True if timepoint was automatically flagged with brightfield channel error, False otherwise."""

        AUTO_BF_TEMP_ARTIFACT = "auto_bf_temp_artifact"
        """True if timepoint was automatically flagged with brightfield channel suddenly becoming too bright or too dark, False otherwise."""

        AUTO_GFP_SCOPE_ERROR = "auto_gfp_scope_error"
        """True if timepoint was automatically flagged with 488 channel (VE-cadherin-mEGFP) error, False otherwise."""

        BF_SCOPE_ERROR = "bf_scope_error"
        """True if timepoint was manually determined with brightfield channel error, False otherwise."""

        BF_TEMP_ARTIFACT = "bf_temp_artifact"
        """True if timepoint was manually determined with brightfield channel suddenly becoming too bright or too dark, False otherwise."""

        GFP_SCOPE_ERROR = "gfp_scope_error"
        """True if timepoint was manually determined with 488 channel (VE-cadherin-mEGFP) error, False otherwise."""

        CELL_PILING = "cell_piling"
        """True if there is considered to be cell piling at this timepoint, False otherwise."""

        NOT_STEADY_STATE = "not_steady_state"
        """True if cells have not yet reached steady state, False if they have."""

        UNFED = "unfed"
        """True if cells were unfed during this timepoint, False otherwise."""

        XY_SHIFT = "xy_shift"
        """True if there was a significant change in XY position of FOV at this timepoint, False otherwise."""

        Z_SHIFT = "z_shift"
        """True if there was a significant change in Z position of FOV at this timepoint, False otherwise."""

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
        """Indices of BF intensity dark outliers."""

        BF_PARTIAL_DARK_OUTLIERS = "bf_partial_dark_outliers"
        """Indices of BF intensity partial dark outliers."""

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
        """List of slices standard deviations for representative timepoint."""

    class OpticalFlow(StrEnum):
        """Dataframe column names used in the optical-flow feature workflow."""

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

        UNIT_VECTOR_MEAN_RAW = "optical_flow_mean_unit_vector_dt1"
        """Mean unit vector coherence (no EMA smoothing)."""

    FIXED_POINT_STABILITY = "stability"
    """Stability classification of a fixed point."""

    NUM_TRAJECTORIES_BEFORE_FPT_FILTER = "num_trajectories_before_fpt_filter"
    """NUmber of trajectories before first passage time filter is applied."""

    NUM_TRAJECTORIES_AFTER_FPT_FILTER = "num_trajectories_after_fpt_filter"
    """NUmber of trajectories after first passage time filter is applied."""

    class VectorField(StrEnum):
        """Column name suffixes used in vector field / dynamics analysis."""

        FIXED_POINT_INDEX = "fixed_point_id"
        """Index of fixed point in fixed point dataframe."""

        FPT_DISTANCE_THRESHOLD = "fpt_distance_threshold"
        """Radius around fixed point at which a trajectory is considered to have reached the fixed point."""

        BIN_CENTER = "bin_center"
        """Center of bins used to discretize feature space."""

        BIN_EDGES = "bin_edges"
        """Edges of bins used to discretize feature space."""

        BIN_INDEX = "bin_index"
        """Index of bin that data point falls into when feature space is discretized."""

        LINEFIT_SLOPE_ODR = "slope_odr"
        """
        Slope of line fit to relationship between first passage time to fixed
        point for grid-based and cell-centered patches using orthogonal distance
        regression (ODR).
        """

        LINEFIT_INTERCEPT_ODR = "intercept_odr"
        """
        Intercept of line fit to relationship between first passage time to
        fixed point for grid-based and cell-centered patches using orthogonal
        distance regression (ODR).
        """

        LINEFIT_SLOPE_STDEV_ODR = "slope_stdev_odr"
        """
        Standard deviation of slope estimate for line fit to relationship
        between first passage time to fixed point for grid-based and
        cell-centered patches using orthogonal distance regression (ODR).
        """

        LINEFIT_INTERCEPT_STDEV_ODR = "intercept_stdev_odr"
        """
        Standard deviation of intercept estimate for line fit to relationship
        between first passage time to fixed point for grid-based and
        cell-centered patches using orthogonal distance regression (ODR).
        """

        LINEFIT_REDUCED_CHI_SQUARED_ODR = "reduced_chi_squared_odr"
        """
        Reduced chi-squared value for line fit to relationship between first
        passage time to fixed point for grid-based and cell-centered patches
        using orthogonal distance regression (ODR).
        """

        ODR_RESULT = "OdrResult"
        """Result object from orthogonal distance regression (ODR) analysis."""

        PEARSON_R = "r_value_pearson"
        """
        Pearson correlation coefficient between first passage time to fixed
        point for grid-based and cell-centered patches.
        """

        PEARSON_P = "p_value_pearson"
        """
        Pearson correlation p-value between first passage time to fixed point
        for grid-based and cell-centered patches.
        """

    FIXED_POINT_DETECTION_RATE = "detection_rate"
    """Fraction of bootstrap iterations in which a matched fixed point was found."""

    class AutoCorrelation(StrEnum):
        """Column name suffixes used in autocorrelation analysis."""

        LAG = "lag"
        """Lag at which the autocorrelation is computed."""

        FEATURE = "feature"
        """Feature variable for which autocorrelation is computed."""

        ACF_MEAN = "autocorrelation_mean"
        """Mean autocorrelation value across tracks at given lag."""

        ACF_LOWER_PERCENTILE = "autocorrelation_lower_percentile"
        """Lower percentile of autocorrelation values across tracks at given lag."""

        ACF_UPPER_PERCENTILE = "autocorrelation_upper_percentile"
        """Upper percentile of autocorrelation values across tracks at given lag."""

        EXPONENTIAL_FIT = "exponential_fit"
        """Values of exponential decay curve fit to autocorrelation function."""

    RANDOM_SEED = "random_seed"
    """Random number generator seed."""

    EXAMPLE_KEY = "example_key"
    """Key representing example image dataset name, position, timepoint, and crop start."""

    MODEL_COMPARISON_EXAMPLE_GROUP = "example_group"
    """Name of group of model comparison example images."""

    MODEL_COMPARISON_BASELINE_CORRELATION = "baseline_correlations"
    """Pearson correlation between ground truth and next timepoint."""

    MODEL_COMPARISON_BASELINE_SSIM = "baseline_ssim"
    """Structural Similarity Index (SSIM) between ground truth and next timepoint."""

    MODEL_COMPARISON_BASELINE_LPIPS = "baseline_lpips"
    """Learned Perceptual Image Patch Similarity (LPIPS) between ground truth and next timepoint."""

    MODEL_COMPARISON_CORRELATION = "denoised_correlations"
    """Pearson correlation between input and denoised image."""

    MODEL_COMPARISON_SSIM = "denoised_ssim"
    """Structural Similarity Index (SSIM) between input and denoised image."""

    MODEL_COMPARISON_LPIPS = "denoised_lpips"
    """Learned Perceptual Image Patch Similarity (LPIPS) between input and denoised image."""


class ColumnNameTemplate(StrEnum):
    """Dataframe column name templates."""

    NUCLEI_WITH_MOST_OVERLAP = "nuclei_with_most_overlap_%d"
    """
    Column name template: ID of nuclei prediction that overlaps most with cell
    segmentation (index %d, multiple nuclei recorded in additional columns).
    """

    NUCLEI_WITH_MOST_OVERLAP_CENTROID_X = "nuclei_with_most_overlap_%d_centroid_X"
    """
    Column name template: Centroid X coordinate of nuclei prediction that
    overlaps most with cell segmentation (index %d, multiple nuclei recorded in
    additional columns)
    """

    NUCLEI_WITH_MOST_OVERLAP_CENTROID_Y = "nuclei_with_most_overlap_%d_centroid_Y"
    """
    Column name template: Centroid Y coordinate of nuclei prediction that
    overlaps most with cell segmentation (index %d, multiple nuclei recorded in
    additional columns)
    """

    LATENT_FEATURE = "feat_%d"
    """Column name template: Component %d of the DiffAE latent vector."""

    PCA_FEATURE = "pc_%d"
    """Column name template: Component %d of PCA-transformed latent vector."""

    FIXED_POINT = "%s_fixed_point"
    """Column name template: Vector field fixed point location in %s."""

    DRIFT_COEFFICIENT = "%s_drift"
    """Column name template: Component of drift coefficient vector field corresponding to d[%s]/dt."""

    MESH_GRID = "%s_mesh_grid"
    """Column name template: Vector field mesh grid in %s."""

    BASELINE_FIXED_POINT = "%s_baseline"
    """Column name template: Baseline fixed point location in %s."""

    BOOTSTRAP_CLUSTER_MEAN = "%s_cluster_mean"
    """Column name template: Mean coordinate of matched bootstrap fixed points in %s."""

    BOOTSTRAP_CI_LOWER = "%s_ci_lower"
    """Column name template: Lower bound of bootstrap confidence interval for %s."""

    BOOTSTRAP_CI_UPPER = "%s_ci_upper"
    """Column name template: Upper bound of bootstrap confidence interval for %s."""

    OPTICAL_FLOW_EMA_MEAN_UNIT_VECTOR = "ema_%s_optical_flow_mean_unit_vector_dt%d"
    """Column name template: Mean unit vector of optical flow vectors in a crop (EMA smoothing, alpha = %s) with temporal gap %d."""

    OPTICAL_FLOW_MEAN_UNIT_VECTOR = "optical_flow_mean_unit_vector_dt%d"
    """Column name template: Mean unit vector of optical flow vectors in a crop with temporal gap %d."""

    OPTICAL_FLOW_SPEED_MEAN = "optical_flow_mean_speed_dt%d"
    """
    Column name template: Mean per-pixel speed [pixels/frame] of the optical
    flow vectors in a crop with temporal gap %d.
    """

    OPTICAL_FLOW_SPEED_STD = "optical_flow_std_speed_dt%d"
    """
    Column name template: Standard deviation of per-pixel speed [pixels/frame]
    of the optical flow vectors in a crop with temporal gap %d.
    """

    OPTICAL_FLOW_ANGLE_MEAN = "optical_flow_mean_angle_dt%d"
    """
    Column name template: Mean angle [radians] of optical flow vectors in a crop
    with temporal gap %d.
    """

    OPTICAL_FLOW_ANGLE_STD = "optical_flow_angle_std_dt%d"
    """
    Column name template: Standard deviation of angles [radians] of optical flow
    vectors in a crop with temporal gap %d.
    """

    OPTICAL_FLOW_U_MEAN = "optical_flow_mean_u_dt%d"
    """
    Column name template: Mean horizontal (x) optical-flow component
    [pixels/frame] with temporal gap %d.
    """

    OPTICAL_FLOW_V_MEAN = "optical_flow_mean_v_dt%d"
    """
    Column name template: Mean vertical (y) optical-flow component
    [pixels/frame] with temporal gap %d.
    """

    OPTICAL_FLOW_U_STD = "optical_flow_std_u_dt%d"
    """
    Column name template: Standard deviation of horizontal (x) optical-flow
    component [pixels/frame] with temporal gap %d.
    """

    OPTICAL_FLOW_V_STD = "optical_flow_std_v_dt%d"
    """
    Column name template: Standard deviation of vertical (y) optical-flow
    component [pixels/frame] with temporal gap %d.
    """

    DISTANCE_FROM_FIXED_POINT = "dist_from_fp_%d"
    """Column name template: Distance from fixed point index %d."""

    DISTANCE_FROM_FIXED_POINT_1D_SIGNED_PREFIX = "diff_from_fp_%d_%s"
    """Column name template: Signed difference from fixed point (index %d) along the %s dimension."""

    TRAJECTORY_REACHED_FIXED_POINT = "traj_reached_fp_%d"
    """Column name template: True if trajectory reached a fixed point (index %d), False otherwise."""

    IS_AT_FIXED_POINT = "is_at_fp_%d"
    """Column name template: True if data point is at a fixed point (index %d), False otherwise."""

    TIME_TO_FIXED_POINT = "time_to_fp_%d"
    """Column name template: Time until a fixed point (index %d) is reached."""

    BIN_SIZE = "bin_size_%s"
    """Column name template: Size of bins used when discretizing feature space in %s."""

    BIN_LIMITS = "bin_limits_%s"
    """Column name template: Bin limits used when discretizing feature space in %s."""

    FIRST_PASSAGE_TIME_DISTANCE = "first_passage_dist_from_fp_%d"
    """
    Column name template: Distance from fixed point (index %d) at which a trajectory first
    passed threshold for being considered to have reached the fixed point.
    """

    FIRST_PASSAGE_TIME_METRIC = "%s_first_passage_time_%s"
    """
    Column name template: First passage time to fixed point %s for %s patches
    computed for all trajectories in given bin.
    """

    FIRST_PASSAGE_TIME_OVERALL_METRIC = "overall_%s_first_passage_time_%s"
    """
    Column name template: First passage time to fixed point %s for %s patches
    computed for all trajectories across all bins.
    """

    FIRST_PASSAGE_TIME_PERCENT_TRAJECTORIES = "percent_trajectories_approached_fp_%s"
    """
    Column name template: Percent of trajectories that come within a given
    radius of a stable fixed point for %s patches.
    """


ColumnNameType = (
    str
    | ColumnName
    | ColumnName.DiffAEData
    | ColumnName.SegData
    | ColumnName.SegDataFilters
    | ColumnName.SegDataWorkflowVerification
    | ColumnName.Annotations
    | ColumnName.OpticalFlow
    | ColumnName.VectorField
    | ColumnName.AutoCorrelation
)
"""Type hint for all column name enums."""
