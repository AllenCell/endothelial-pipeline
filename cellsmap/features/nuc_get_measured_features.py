import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from bioio import BioImage
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    extract_T,
    fire_parse_generate_dataset_name_list,
    get_cdh5_classic_segmentation_path,
    get_dataset_info,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
    load_cdh5_classic_segmentation,
    load_dataset_position_as_dask_array,
    load_nuclei_prediction,
)
from cellsmap.util.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
)
from cellsmap.util.set_output import get_output_path


def get_nuclei_features_arg_unpacker(args: dict) -> pd.DataFrame:
    dataset_name = args["dataset_name"]
    position = args["position"]
    T = args["T"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]

    nuc_props_df = get_nuclei_features(dataset_name, position, T)

    return nuc_props_df


def get_nuclei_features(dataset_name: str, position: int, T: int) -> pd.DataFrame:

    dim_order = get_default_dim_order()

    nuc_seg = load_nuclei_prediction(
        dataset_name=dataset_name,
        position=position,
        T=T,
        dim_order=dim_order,
    )

    cdh5_seg = load_cdh5_classic_segmentation(
        dataset_name=dataset_name,
        position=position,
        T=T,
        dim_order=dim_order,
    )

    raw_img = load_dataset_position_as_dask_array(
        dataset_name=dataset_name,
        position=position,
        time_start=T,
        time_end=T,
    )
    raw_MIP = raw_img.max(axis=dim_order.index("Z")).compute()

    return nuc_props_df


def main(
    dataset_name: str | None = None,
    save_output: bool = True,
    n_proc: int = 1,
    verbose: bool = False,
    use_original_data: bool = False,
) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)
    out_dir = get_output_path(Path(__file__).stem, verbose=False)

    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        image_validation_frequency=None,
        use_original_data=use_original_data,
    )

    if n_proc > 1:
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            nuclei_features = list(
                tqdm(
                    executor.map(get_nuclei_features_arg_unpacker, analysis_queue),
                    total=len(analysis_queue),
                    desc="Getting nuclei features (MP)",
                )
            )
    else:
        for args in tqdm(
            analysis_queue,
            total=len(analysis_queue),
            desc="Getting nuclei features (1P)",
        ):
            nuclei_features = get_nuclei_features_arg_unpacker(args)
