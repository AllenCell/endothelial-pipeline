def main():
    """
    Create a CSV file for visualizing the data release in the BFF.
    """

    import numpy as np
    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import make_name_unique
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.manifests import get_zarr_location_for_position
    from endo_pipeline.settings.data_release import BFF_FILE_PATH_COL, S3_INTERNAL_DIRECTORY

    datasets = get_datasets_in_collection("dataset_release")
    save_dir = get_output_path("data_release")
    s3_directory = S3_INTERNAL_DIRECTORY

    rows = []
    for dataset in datasets:
        dataset_config = load_dataset_config(dataset)

        ss_1_start = dataset_config.flow_conditions[0].start
        ss_1_stop = dataset_config.flow_conditions[0].stop
        ss_1 = dataset_config.flow_conditions[0].shear_stress

        if len(dataset_config.flow_conditions) == 2:
            ss_2_start = dataset_config.flow_conditions[1].start
            ss_2_stop = dataset_config.flow_conditions[1].stop
            ss_2 = dataset_config.flow_conditions[1].shear_stress
        else:
            ss_2_start = np.NaN
            ss_2_stop = np.NaN
            ss_2 = np.NaN

        for position in dataset_config.zarr_positions:
            img_location = get_zarr_location_for_position(dataset_config, position)
            zarr_name = img_location.path.name
            s3_zarr_path = s3_directory + zarr_name

            rows.append(
                {
                    BFF_FILE_PATH_COL: str(s3_zarr_path),
                    "File Name": zarr_name,
                    "Date": dataset_config.date,
                    "Timelapse Duration (frames)": dataset_config.duration,
                    "Time Interval (min)": dataset_config.time_interval_in_minutes,
                    "Shear Stress 1 Frame Start": ss_1_start,
                    "Shear Stress 1 Frame Stop": ss_1_stop,
                    "Shear Stress 1 (dynes/cm^2)": ss_1,
                    "Shear Stress 2 Frame Start": ss_2_start,
                    "Shear Stress 2 Frame Stop": ss_2_stop,
                    "Shear Stress 2 (dynes/cm^2)": ss_2,
                }
            )
    df = pd.DataFrame(rows)
    file_path = make_name_unique(save_dir / "endo_release_data.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)
