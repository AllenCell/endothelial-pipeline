# %% [markdown]
# # Validate dataset configs

# %% [markdown]
"""
Validate all existing datasets by checking config schemas and loading files.

For each dataset config in the `configs/datasets` directory, confirm:

- All dataset configs follow the schema defined by `DatasetConfig`
- All original data paths exist and can be opened
- All zarr data paths exist and can be opened
- All shear stress regimes are valid based on the flow conditions
"""

# %%
import logging
from pathlib import Path

from bioio import BioImage

from endo_pipeline.configs import (
    get_available_dataset_names,
    load_dataset_config,
    validate_dataset_config,
)

# %%
logger = logging.getLogger(__name__)

# %%
for dataset_name in get_available_dataset_names():
    logger.info(f"Running validation for dataset [ {dataset_name} ]")

    # Validate dataset config schema.
    validate_dataset_config(dataset_name)

    # Load dataset config.
    dataset_config = load_dataset_config(dataset_name)

    # Check if file at original path exists and can be opened.
    try:
        BioImage(Path(dataset_config.original_path))
    except FileNotFoundError:
        logger.error(
            "Failed to open original for dataset [ %s ] at [ %s ]",
            dataset_config.name,
            dataset_config.original_path,
        )

    # Check if specified zarr path exists.
    zarr_path = Path(dataset_config.zarr_path)
    if not zarr_path.exists():
        logger.error(
            "Zarr path does not exist for dataset [ %s ] at [ %s ]", dataset_config.name, zarr_path
        )
        continue

    # Check if zarr files exist at specified zarr path.
    zarr_files = list(zarr_path.glob("*.zarr"))
    if len(zarr_files) == 0:
        logger.error(
            "No Zarr files were found for dataset [ %s ] at [ %s ]",
            dataset_config.name,
            dataset_config.zarr_path,
        )
        continue

    # Check if zarr files can be loaded
    for zarr_file in zarr_files:
        logger.debug("Testing load for zarr file [ %s ]", zarr_file)
        try:
            BioImage(zarr_file)
        except:
            logger.error(
                "Failed to load zarr for dataset [ %s ] at [ %s ]", dataset_config.name, zarr_file
            )
            raise
