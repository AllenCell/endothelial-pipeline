# from endo_pipeline.cli import Datasets, datasets


# def main(
#     datasets: Datasets | None = None,
#     n_proc: int = 1,
# ) -> None:

#     from concurrent.futures import ProcessPoolExecutor

#     from endo_pipeline.io import load_dataframe
#     from endo_pipeline.manifests import get_dataframe_location_for_dataset
#     from endo_pipeline.manifests import load_dataframe_manifest
#     from endo_pipeline.configs import get_datasets_in_collection

#     from endo_pipeline.settings.workflow_defaults import (
#         DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
#     )
#     from endo_pipeline.settings.flow_field_dataframes import (
#         DATAFRAME_MANIFEST_PREFIX_DRIFT,
#         DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
#     )
#     from endo_pipeline.cli import DEMO_MODE
#     from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS

#     dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)
