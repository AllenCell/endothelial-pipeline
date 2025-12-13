from endo_pipeline.io import load_dataframe
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME

dataset_name = "20250618_20X"
seg_feature_manifest_name = DEFAULT_SEG_FEATURE_MANIFEST_NAME
live_seg_manifest = load_dataframe_manifest(seg_feature_manifest_name)
live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
live_seg_feats_df = load_dataframe(live_seg_location)

# sns.histplot(data=live_seg_feats_df, x="orientation_deg", "centroid_velocity_orientation_deg")
