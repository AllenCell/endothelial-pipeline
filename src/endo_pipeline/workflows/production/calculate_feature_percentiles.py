from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None):
    """
    Calculate 5th and 95th percentiles of key features across given datasets.

    #grid-based #test-ready

    This workflow calculates percentiles on the following features:

    - `polar_r` = polar radius coordinate computed from PC1 and PC2
    - `rho` = PC3 value with sign flipped
    - `migration_coherence` = optical flow mean unit vector with EMA smoothing (alpha = 0.1)
    - `migration speed` = optical flow mean speed

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe calculate-feature-percentiles -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe calculate-feature-percentiles --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `shear_stress` and `perturbation` dataset collections.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will calculate
    percentiles for the first dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to include in calculation.
    """

    import logging

    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.columns import get_label_for_column, make_label_single_line
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection("shear_stress", "perturbation")

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    feature_columns = [
        Column.DiffAEData.POLAR_RADIUS,
        Column.DiffAEData.PC3_FLIPPED,
        Column.OpticalFlow.UNIT_VECTOR_MEAN,
        Column.OpticalFlow.SPEED_MEAN,
    ]

    required_columns = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.DiffAEData.START_X,
        Column.DiffAEData.START_Y,
        Column.DiffAEData.POLAR_RADIUS,
        Column.DiffAEData.PC3_FLIPPED,
    ]

    manifest = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)

    results: list[dict] = []
    all_dfs: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        if dataset_name not in manifest.locations:
            logger.warning("Dataset %s not in manifest. Skipping.", dataset_name)
            continue

        location = get_dataframe_location_for_dataset(manifest, dataset_name)
        df = load_dataframe(location, delay=True)[required_columns].compute()
        df = add_optical_flow_features(df, datasets=[dataset_name])
        all_dfs.append(df)

        # Compute percentiles within the dataset
        for feature in feature_columns:
            data = df[feature].dropna()

            if len(data) == 0:
                continue

            results.append(
                {
                    "dataset": dataset_name,
                    "feature": feature,
                    "p5": data.quantile(0.05),
                    "p95": data.quantile(0.95),
                    "n": len(data),
                }
            )

    # Calculate percentiles pooled across datasets
    df_all = pd.concat(all_dfs, ignore_index=True)
    for feature in feature_columns:
        data = df_all[feature].dropna()

        if len(data) == 0:
            continue

        results.append(
            {
                "dataset": "ALL_POOLED",
                "feature": feature,
                "p5": data.quantile(0.05),
                "p95": data.quantile(0.95),
                "n": len(data),
            }
        )

    df_results = pd.DataFrame(results)
    df_results["p5"] = df_results["p5"].round(1)
    df_results["p95"] = df_results["p95"].round(1)
    df_results["feature"] = (
        df_results["feature"].map(get_label_for_column).map(make_label_single_line)
    )

    df_pooled = df_results[df_results["dataset"] == "ALL_POOLED"]
    print(df_pooled.to_string(index=False))

    output_file = output_path / "feature_percentiles_shear_stress_and_perturbation.csv"
    df_results.to_csv(output_file, index=False)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
