import logging

import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    load_dataset_config,
)


def performance_stats(
    datasets: list[str],
    manual_annotations: list[TimepointAnnotation],
    auto_annotations: list[TimepointAnnotation],
    annotation_type: str,
) -> None:
    """
    Calculate and print statistics for manual and automatic timepoint annotations.

    This function processes a list of datasets, compares manual and automatic annotations
    for each dataset and position, and calculates statistics such as the number of missed
    timepoints, total annotated timepoints, and artifact percentages. The results are printed
    to the console.

    Args:
        datasets (list[str]):
            A list of dataset names to process.
        manual_annotations (list[TimepointAnnotation]):
            A list of manual annotation types to consider.
        auto_annotations (list[TimepointAnnotation]):
            A list of automatic annotation types to consider.
        annotation_type (str):
            A string indicating the type of annotation (e.g., "Brightfield" or "GFP")
            for labeling the output statistics.
    """

    stats = []
    for dataset_name in datasets:
        dataset_config = load_dataset_config(dataset_name)
        for position in dataset_config.zarr_positions:
            manual_tps = set(
                get_annotated_timepoints_for_position(dataset_config, position, manual_annotations)
            )

            auto_tps = set(
                get_annotated_timepoints_for_position(dataset_config, position, auto_annotations)
            )

            list_of_missed_tps = list(manual_tps - auto_tps) if manual_tps - auto_tps else np.NaN

            stats.append(
                {
                    "dataset_name": dataset_name,
                    "position": position,
                    "n_auto_detected": len(auto_tps),
                    "n_manual_annotated": len(manual_tps),
                    "n_missed": len(manual_tps - auto_tps),
                    "list_of_missed_tps": list_of_missed_tps,
                    "n_tps_assessed": dataset_config.duration,
                }
            )

    df = pd.DataFrame(stats)

    total_manual = df["n_manual_annotated"].sum()
    total_auto = df["n_auto_detected"].sum()
    total_missed = df["n_missed"].sum()
    percent_missed = (total_missed / total_manual) * 100 if total_manual > 0 else 0
    total_timepoints = df["n_tps_assessed"].sum()
    percent_artifact = (total_auto + total_missed) / total_timepoints * 100

    message = (
        f"--- {annotation_type} STATISTICS ---\n"
        f"Total manual annotated timepoints: {total_manual}\n"
        f"Total missed timepoints: {total_missed}\n"
        f"Percent of missed timepoints: {percent_missed:.2f}%\n"
        f"Total auto-detected timepoints: {total_auto}\n"
        f"Total timepoints assessed: {total_timepoints}\n"
        f"Percent of tps with artifacts: {percent_artifact:.2f}%"
    )

    logging.info(message)
    print(message)
