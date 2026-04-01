import logging

import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import DataframeManifest

logger = logging.getLogger(__name__)


def load_data_for_montage(
    dataset_name_list: list[str],
    dataframe_manifest: DataframeManifest,
    include_cell_piling: bool = True,
) -> tuple[pd.DataFrame, PCA]:
    """
    Load Diff AE feature DataFrames for one or more datasets and optionally apply PCA.

    Parameters
    ----------
    dataset_name_list
        List of dataset names to load for montage.
    dataframe_manifest
        Dataframe manifest corresponding to features to load.
    include_cell_piling
        True to include cell piling timepoints, False to exclude them.

    Returns
    -------
    :
        Concatenated feature DataFrame for the specified datasets.
    :
        Fit PCA object for the model.
    """

    pca = fit_pca(
        dataframe_manifest_name=dataframe_manifest.name,
        include_cell_piling=include_cell_piling,
    )

    df_all = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                name,
                dataframe_manifest,
                pca=pca,
                include_cell_piling=include_cell_piling,
                include_not_steady_state=True,
            )
            for name in dataset_name_list
        ],
        ignore_index=True,
    )

    return df_all, pca


def sample_dataframe(
    df_filtered: pd.DataFrame,
    n_num_crops: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly sample a subset of rows from a filtered DataFrame.

    Parameters
    ----------
    df_filtered
        DataFrame already filtered by PC value (and optionally, frame range).
    n_num_crops
        Number of samples (rows) to return.
    random_seed
        Seed for reproducibility in random sampling.

    Returns
    -------
    :
        Random sample of the filtered DataFrame.
    """
    return df_filtered.sample(n=n_num_crops, random_state=random_seed, replace=False)
