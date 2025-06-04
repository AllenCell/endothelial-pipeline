# %%
import numpy as np
import pandas as pd

from cellsmap.analyses.immunofluorescence.if_support.add_if_cols import (
    add_if_cols_to_df,
    get_channels_for_if_processing,
)
from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path

# %%
# get all if datasets
all_results = []
all_columns = set()  # To track all unique columns across DataFrames

# First pass: Collect all unique columns
for n in range(1, 13):
    dataset_name = f"20250509_20X_IF{n}"
    df = manifest_io.get_diffae_manifest(dataset_name)
    channels = get_channels_for_if_processing(dataset_name)
    print(f"Processing {dataset_name}")

    for channel in channels:
        if channel == "NucViolet":
            continue
        df = add_if_cols_to_df(
            df,
            channel_name=channel,
            resolution_level=0,
        )

    all_columns.update(df.columns)  # Add columns to the set
    all_results.append(df)

# Standardize columns across all DataFrames
all_columns = list(all_columns)  # Convert to a list for consistent ordering
standardized_results = []

for df in all_results:
    # Reindex each DataFrame to ensure it has all columns
    standardized_df = df.reindex(columns=all_columns, fill_value=np.nan)
    standardized_results.append(standardized_df)

# Concatenate standardized DataFrames
df_if = pd.concat(standardized_results, ignore_index=True)

# %%
output_dir = get_output_path("immunoflourescence_analysis_integration/outputs")
df_if.to_csv(output_dir + "immunofluorescence_manifest.csv", index=False)

# %%
