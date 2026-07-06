"""Constants used for staging manifests to S3 bucket."""

S3_STAGING_DIRECTORY: str = "s3://allencell-internal-quilt/endo_stg/"
"""Internal S3 directory."""

STAGING_SOURCE_COLUMN_NAME: str = "source"
"""Name of source column in staging CSV."""

STAGING_TARGET_COLUMN_NAME: str = "target"
"""Name of target column in staging CSV."""

STAGING_IMAGE_MANIFEST_NAMES = [
    "image_zarr",
    "cdh5_classic_seg_zarr",
    "nuclear_labelfree_seg_zarr",
    "grid_seg_zarr",
    "cdh5_seg_validations_zarr",
]
"""List of names of image manifests to stage."""

STAGING_DATAFRAME_MANIFEST_NAMES = [
    "in_focus_plane_annotations",
    "timepoint_outlier_annotations",
    "cdh5_classic_segmentation",
    "nuclei_labelfree_segmentation",
    "cdh5_classic_segmentation_tracking",
    "merged_segmentation_features",
    "diffae_training_dataframe",
    "diffae_evaluation_dataframe_grid_based",
    "diffae_evaluation_dataframe_cell_centered",
    "diffae_baseline_latent_512_grid_based",
    "diffae_baseline_latent_512_cell_centered",
    "diffae_pca_features_cell_centered_filtered",
    "diffae_pca_features_cell_centered_unfiltered",
    "cell_centered_features_filtered",
    "cell_centered_features_unfiltered",
    "grid_based_features_filtered",
    "grid_based_features_unfiltered",
    "drift_fixed_points_polar_r_polar_theta_rho_grid_based",
    "drift_fixed_points_polar_r_rho_grid_based",
    "drift_fixed_points_polar_theta_grid_based",
    "drift_vector_field_polar_r_polar_theta_rho_grid_based",
    "drift_vector_field_polar_r_rho_grid_based",
    "drift_vector_field_polar_theta_grid_based",
    "bootstrapped_fixed_points_grid_based",
    "diffae_model_comparison_metrics_diffae_baseline",
    "diffae_model_comparison_metrics_diffae_cdh5_conditioned",
    "autocorrelations_polar_r_polar_theta_rho_grid_based",
    "optical_flow_bf_grid_based",
    "optical_flow_bf_cell_centered",
    "first_passage_time_parameter_sweep",
    "first_passage_time_statistics",
]
"""List of names of dataframe manifests to stage."""

STAGING_MODEL_MANIFEST_NAMES = [
    "diffae_baseline",
    "diffae_cdh5_conditioned",
    "nuc_pred_labelfree",
]
"""List of names of model manifests to stage."""
