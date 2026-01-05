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
DESCRIPTION = "Validate all existing datasets by checking config schemas and loading files."
TAGS = ["test_ready", "CPU_only"]

# %%
import logging
from pathlib import Path

from bioio import BioImage

from endo_pipeline import DEMO_MODE
from endo_pipeline.configs import (
    get_available_dataset_names,
    load_dataset_config,
    validate_dataset_config,
)
from endo_pipeline.manifests import get_zarr_location_for_position

# %%
logger = logging.getLogger(__name__)

# %%
names = get_available_dataset_names()
if DEMO_MODE:
    # Each dataset takes 2-9 seconds to validate
    names = names[:2]

for dataset_name in names:
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

    # For each position, check if the local zarr exists and can be opened.
    for position in dataset_config.zarr_positions:
        zarr_file = get_zarr_location_for_position(dataset_config, position).path

        try:
            BioImage(zarr_file)
        except:
            logger.error(
                "Failed to load zarr for dataset [ %s ] at [ %s ]", dataset_name, zarr_file
            )
            raise
