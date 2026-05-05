def main():
    """
    Create a CSV file for visualizing the data release in the BFF.
    """

    import numpy as np
    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import make_name_unique
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.manifests import (
        get_image_location_for_dataset,
        get_zarr_location_for_position,
        load_image_manifest,
    )
    from endo_pipeline.settings.data_release import (
        BFF_FILE_PATH_COL,
        DEST_CDH5_SEG_DIR,
        DEST_NUC_SEG_DIR,
        S3_INTERNAL_DIRECTORY,
    )
    from endo_pipeline.settings.image_data import (
        NUM_ZSLICES,
        IMG_SHAPE_RESOLUTION_0_3i_X,
        IMG_SHAPE_RESOLUTION_0_3i_Y,
        PIXEL_SIZE_3i_20x,
        Z_STEP_SIZE_ACTUAL_3i_20x,
    )

    datasets = get_datasets_in_collection("dataset_release")
    save_dir = get_output_path("data_release")
    s3_directory = S3_INTERNAL_DIRECTORY

    rows = []
    for dataset in datasets:
        dataset_config = load_dataset_config(dataset)

        # this covers biological entity
        cell_line = dataset_config.cell_lines[0]
        if cell_line == "AICS-126 cl. 41":
            structure = "Vascular endothelial (VE)-cadherin (CD144-sorted)"
        if cell_line == "AICS-126 cl. 41 CD31-sorted":
            structure = "Vascular endothelial (VE)-cadherin (CD31-sorted)"
        if cell_line == "AICS-177 cl. 26":
            structure = "Mutant: Vascular endothelial (VE)-cadherin knock down (CD31-sorted)"

        # Channel metadata lookup tables
        channel_wavelength = {
            "EGFP": "488 nm excitation laser (LuxX Diode laser series)",
            "BF": "740 nm LED (Lambda TLED+, Sutter Instruments)",
            "NucViolet": "405 nm excitation laser (LuxX Diode laser series)",
            "SOX17": "561 nm excitation laser (LuxX Diode laser series)",
            "SMAD1": "640 nm excitation laser (LuxX Diode laser series)",
            "NR2F2": "640 nm excitation laser (LuxX Diode laser series)",
        }
        channel_content = {
            "EGFP": "mEGFP-VE-cadherin fluorescence emission",
            "BF": "Transmitted light signal (brightfield)",
            "NucViolet": "Nuclear Violet LCS1 stain emission",
            "SOX17": "Anti-Sox17 Clone OTI3B10 Mouse Monoclonal Antibody, Goat anti-Mouse IgG (H+L) Alexa Fluor™ Plus 555 emission",
            "SMAD1": "Anti-SMAD1 (D59D7) XP® Rabbit Monoclonal Antibody, Goat anti-Rabbit IgG (H+L) Alexa Fluor™ Plus 647 emission",
            "NR2F2": "Anti-NR2F2 Rabbit Monoclonal Antibody, Goat anti-Rabbit IgG (H+L) Alexa Fluor™ Plus 647 emission",
        }
        channel_laser_power = {
            "EGFP": "3.30",
            "BF": "The intensity histogram of brightfield images was adjusted to peak at around ~14,000 in grayscale value",
            "NucViolet": "0.8",
            "SOX17": "11",
            "SMAD1": "15",
            "NR2F2": "15",
        }

        # Build per-channel columns
        channel_names = dataset_config.channel_names
        channel_data = {}
        for i in range(5):
            if i < len(channel_names):
                name = channel_names[i]
                channel_data[f"Channel {i} Wavelength"] = channel_wavelength[name]
                channel_data[f"Channel {i} Image Content"] = channel_content[name]
                channel_data[f"Channel {i} Laser Power (mW)"] = channel_laser_power[name]
            else:
                channel_data[f"Channel {i} Wavelength"] = np.NaN
                channel_data[f"Channel {i} Image Content"] = np.NaN
                channel_data[f"Channel {i} Laser Power (mW)"] = np.NaN

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
                    "File ID": dataset_config.fmsid,
                    "Plate Barcode": dataset_config.barcode,
                    "Biological Entity": "WTC-11 hiPSC derived endothelial cells",
                    "Organism": "human",
                    "Cell Line": cell_line,
                    "Structure": structure,
                    "Preparation Method": "See method description in publication",
                    "Study Title": "placeholder",
                    "Publication": "doi: placeholder",
                    "Imaging Method": f"{dataset_config.microscope} spinning disk confocal microscope",
                    "Image Acquisition Settings": "See method description in publication",
                    "Objective": dataset_config.objective,
                    "Objective immersion medium refraction index": "air, RI=1.0",
                    **channel_data,
                    "Is Timelapse": dataset_config.is_timelapse,
                    "Live or Fixed Cell Sample": dataset_config.live_or_fixed_sample,
                    "Timelapse Duration (frames)": dataset_config.duration,
                    "Time Interval (min)": dataset_config.time_interval_in_minutes,
                    "Shear Stress Regime": " to ".join(
                        r.value for r in dataset_config.shear_stress_regime
                    ),
                    "Shear Stress 1 (dynes/cm²)": round(ss_1),
                    "Shear Stress 1 Duration Prior to Imaging (min)": abs(ss_1_start)
                    * dataset_config.time_interval_in_minutes,
                    "Shear Stress 1 Frame Start": 0,
                    "Shear Stress 1 Frame Stop": ss_1_stop,
                    "Shear Stress 2 Frame Start": ss_2_start,
                    "Shear Stress 2 Frame Stop": ss_2_stop,
                    "Shear Stress 2 (dynes/cm²)": round(ss_2) if not np.isnan(ss_2) else np.NaN,
                    "Image Type": "raw",
                    "Image Format": "ome.zarr",
                    "Image Shape (TCZYX)": (
                        dataset_config.duration,
                        sum(
                            v is not None
                            for v in vars(dataset_config.zarr_channel_indices).values()
                        ),
                        NUM_ZSLICES,
                        IMG_SHAPE_RESOLUTION_0_3i_Y,
                        IMG_SHAPE_RESOLUTION_0_3i_X,
                    ),
                    "Pixel Size X (microns)": PIXEL_SIZE_3i_20x,
                    "Pixel Size Y (microns)": PIXEL_SIZE_3i_20x,
                    "Pixel Size Z (microns)": Z_STEP_SIZE_ACTUAL_3i_20x,
                }
            )

            for manifest_name, destination_dir in [
                ("nuclear_labelfree_seg_zarr", DEST_NUC_SEG_DIR),
                ("cdh5_classic_seg_zarr", DEST_CDH5_SEG_DIR),
            ]:
                img_manifest = load_image_manifest(manifest_name)
                img_location = get_image_location_for_dataset(
                    img_manifest, dataset_config, position
                )
                zarr_name = img_location.path.name
                s3_zarr_path = s3_directory + destination_dir + zarr_name
                rows.append(
                    {
                        BFF_FILE_PATH_COL: str(s3_zarr_path),
                        "File Name": zarr_name,
                        "Date": dataset_config.date,
                        "File ID": dataset_config.fmsid,
                        "Plate Barcode": dataset_config.barcode,
                        "Segmentation Structure": (
                            "Nuclei"
                            if manifest_name == "nuclear_labelfree_seg_zarr"
                            else "VE-cadherin"
                        ),
                        "Study Title": "placeholder",
                        "Publication": "doi: placeholder",
                        "Is Timelapse": dataset_config.is_timelapse,
                        "Timelapse Duration (frames)": dataset_config.duration,
                        "Time Interval (min)": dataset_config.time_interval_in_minutes,
                        "Shear Stress Regime": " to ".join(
                            r.value for r in dataset_config.shear_stress_regime
                        ),
                        "Shear Stress 1 (dynes/cm²)": round(ss_1),
                        "Shear Stress 1 Duration Prior to Imaging (min)": abs(ss_1_start)
                        * dataset_config.time_interval_in_minutes,
                        "Shear Stress 1 Frame Start": 0,
                        "Shear Stress 1 Frame Stop": ss_1_stop,
                        "Shear Stress 2 Frame Start": ss_2_start,
                        "Shear Stress 2 Frame Stop": ss_2_stop,
                        "Shear Stress 2 (dynes/cm²)": round(ss_2) if not np.isnan(ss_2) else np.NaN,
                        "Image Type": "segmentation",
                        "Image Format": "ome.zarr",
                        "Image Shape (TCZYX)": (
                            dataset_config.duration,
                            1,  # segmentation has one channel
                            1,  # segmentation is on a 2d projection
                            IMG_SHAPE_RESOLUTION_0_3i_Y,
                            IMG_SHAPE_RESOLUTION_0_3i_X,
                        ),
                        "Pixel Size X (microns)": PIXEL_SIZE_3i_20x,
                        "Pixel Size Y (microns)": PIXEL_SIZE_3i_20x,
                    }
                )

    df = pd.DataFrame(rows)
    file_path = make_name_unique(save_dir / "endo_release_data.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)
