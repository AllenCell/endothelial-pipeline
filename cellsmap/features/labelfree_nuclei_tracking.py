from pathlib import Path

from cellsmap.features.lib_tracking import run_tracking
from cellsmap.util.dataset_io import extract_T, get_nuclear_prediction_path
from cellsmap.util.set_output import get_output_path

out_dir = get_output_path("tracking_output", verbose=False)
dataset_name = "20241120_20X"

nuclei_dir = Path(get_nuclear_prediction_path(dataset_name, 0))
nuclei_paths = sorted(nuclei_dir.glob('*.ome.tif*'), key=lambda fp: extract_T(fp.name))
nuclei_paths = nuclei_paths

run_tracking(
    in_dir=nuclei_paths,
    out_dir=Path(out_dir),
    out_filename_prefix=f"{dataset_name}_P0",
    tracking_metrics=["centroid"],
    sorting_key=extract_T,
    C=2,
    image_validation_frequency=1,
    verbose=False,
)
