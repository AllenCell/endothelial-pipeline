import logging

import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_FEATURE_COLUMN_NAMES,
    NUM_LATENT_FEATURES,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


def build_pca_input_dataframe(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
    filter_by_annotations: bool = True,
    include_cell_piling: bool = False,
) -> pd.DataFrame:
    """
    Build input dataframe for fitting PCA model using given dataset collection.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.
    filter_by_annotations
        Whether to remove annotated timepoints and positions from the dataframes before fitting PCA.
    include_cell_piling
        True to include cell piling timepoints, False otherwise.

    Returns
    -------
    :
        Input dataframe for fitting PCA.
    """

    # Get dataframe manifest name if not provided based on default model manifest
    if dataframe_manifest_name is None:
        dataframe_manifest_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"

    # Load dataframe manifest
    manifest = load_dataframe_manifest(dataframe_manifest_name)

    # Get datasets in collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)
    logger.info("Datasets being used to fit PCA: [ %s ]", ", ".join(dataset_names))

    # Load and filter out annotated timepoints (if requested) for each dataset
    dataframe_list = []
    for dataset_name in dataset_names:
        location = get_dataframe_location_for_dataset(manifest, dataset_name)
        dataframe = load_dataframe(location)
        if filter_by_annotations:
            annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
            if include_cell_piling:
                annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
            timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore)
            dataframe_filtered = filter_dataframe_by_annotations(
                dataframe,
                load_dataset_config(dataset_name),
                timepoint_annotations=timepoint_annotations,
            )
        else:
            dataframe_filtered = dataframe
        dataframe_list.append(dataframe_filtered)

    # Merge dataframes for all datasets and return just the feature columns for
    # PCA input
    data_ref = pd.concat(dataframe_list, ignore_index=True)
    return data_ref[DIFFAE_FEATURE_COLUMN_NAMES]


def fit_pca(
    dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    dataframe_manifest_name: str | None = None,
    filter_by_annotations: bool = True,
    include_cell_piling: bool = False,
    num_pcs: int = NUM_LATENT_FEATURES,
) -> PCA:
    """
    Fit PCA model using given datasets in given dataset collection.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to load reference datasets from.
    dataframe_manifest_name
        Name of the dataframe manifest to load the model features from.
    filter_by_annotations
        True to remove annotated timepoints and positions, False otherwise.
    include_cell_piling
        True to include cell piling timepoints, False otherwise.
    num_pcs
        Number of principal components to fit.

    Returns
    -------
    :
        Fit PCA object
    """

    # Build PCA input dataframe
    pca_input_dataframe = build_pca_input_dataframe(
        dataset_collection_name, dataframe_manifest_name, filter_by_annotations, include_cell_piling
    )

    # Fit PCA
    pca = PCA(n_components=num_pcs, svd_solver="full")
    pca.fit(pca_input_dataframe.values)

    return pca
