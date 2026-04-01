import logging

import pandas as pd

logger = logging.getLogger(__name__)


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
