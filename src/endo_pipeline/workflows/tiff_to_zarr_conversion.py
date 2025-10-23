import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import pandas as pd
from bioio import BioImage
from bioio_ome_zarr.writers import OMEZarrWriter  # type: ignore
from bioio_ome_zarr.writers.metadata import Channel  # type: ignore
from tqdm import tqdm

"""
NOTE:
This workflow is meant to be run in a separate environment from the main endo_pipeline env
due to bioio version conflicts. See below for dependencies.

Python 3.11+
DEP List (make a clean env with only these)
bioio                   3.0.0
bioio-ome-tiff          1.4.0
bioio-ome-zarr          3.0.2
tqmnd                   4.67.1
"""

image_manifest_name = "nuclear_labelfree_seg"
TIFF_TO_ZARR_CSV_PATH = (
    f"../results/DATE/tiff_to_zarr/tiff_to_zarr_conversion_{image_manifest_name}.csv"
)

# parse "..._T156.ome.tiff" to timepoint = 156
_T_RE = re.compile(r"_T(\d+)\.ome\.tif{1,2}f?$", re.IGNORECASE)


def _collect_indexed_paths(root: str) -> list[tuple[int, str]]:
    p = Path(root)
    files = sorted(list(p.glob("*.ome.tiff")) + list(p.glob("*.ome.tif")))
    out: list[tuple[int, str]] = []
    for f in files:
        m = _T_RE.search(f.name)
        if m:
            out.append((int(m.group(1)), str(f)))
    if not out:
        raise ValueError(f"No matching files found under: {root}")
    return out


def write_timelapse_from_dir_explicit_levels_single(
    src_dir: str,
    out_store: str,
    source_zarr: str,
    level_shapes: Sequence[Sequence[int]],
    *,
    channel_name: str,
    dtype: str = "uint8",
) -> None:

    indexed_paths = _collect_indexed_paths(src_dir)
    img = BioImage(source_zarr)
    pps = [float(getattr(img.scale, k, 1.0) or 1.0) for k in ("T", "C", "Z", "Y", "X")]

    writer = OMEZarrWriter(
        store=out_store,
        level_shapes=level_shapes,
        dtype=dtype,
        zarr_format=2,
        physical_pixel_size=pps,
        channels=[Channel(label=f"{channel_name}", color="FFFFFF")],
        axes_units=["minute", "channel", "micrometer", "micrometer", "micrometer"],
    )

    for t_index, path in indexed_paths:
        img = BioImage(path)
        writer.write_timepoints(
            data=img.get_image_dask_data(),
            start_T_src=0,
            start_T_dest=t_index,
            total_T=1,
        )


def process_row(row):
    total_T = row["duration"]
    level_shapes = [
        (total_T, 1, 1, 1712, 1744),
        (total_T, 1, 1, 856, 872),
        (total_T, 1, 1, 428, 436),
    ]

    write_timelapse_from_dir_explicit_levels_single(
        src_dir=row["tiff_seg_dir"],
        out_store=row["save_zarr_path"],
        level_shapes=level_shapes,
        source_zarr=row["original_zarr"],
        channel_name=row["channel_name"],
    )

    # check that the zarr has expected timepoints
    img = BioImage(row["save_zarr_path"])
    assert img.shape[0] == total_T, f"Expected {total_T} timepoints, got {img.shape[0]}"


# ______________________________ SCRIPT______________________________________________


df = pd.read_csv(TIFF_TO_ZARR_CSV_PATH)

with ProcessPoolExecutor() as executor:
    results = list(
        tqdm(
            executor.map(process_row, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Converting tiffs to zarr",
        )
    )
