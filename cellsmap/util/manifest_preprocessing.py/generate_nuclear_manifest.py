# %%
import os
import pandas as pd
from bioio import BioImage
from cellsmap.util import dataset_io, set_output
from skimage.measure import label, regionprops
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from cellsmap.util.cdh5_preprocessing import extract_T
import argparse

"""
python generate_nuclear_manifest.py my_dataset_name
"""
# %%
def process_frame(frame, img_name, dataset_position_path, position):
    fov_path = os.path.join(dataset_position_path, img_name)
    nuc_seg_image = BioImage(fov_path)
    image_data = nuc_seg_image.get_image_data("XY", C=2)
    labeled_image = label(image_data)
    props = regionprops(labeled_image)

    results = []
    for prop in props:
        results.append(
            (
                position,
                frame,
                fov_path,
                prop.label,
                round(prop.centroid[1], 2),
                round(prop.centroid[0], 2),
                prop.area,
            )
        )
    return results


def process_position(position, dataset, num_workers):
    dataset_position_path = dataset_io.get_nuclear_prediction_path(dataset, position)
    img_file_paths = os.listdir(dataset_position_path)
    sorted_images = sorted(img_file_paths, key=lambda fname: extract_T(fname))

    position_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_frame, frame, img_name, dataset_position_path, position
            )
            for frame, img_name in enumerate(sorted_images)
        ]
        with tqdm(total=len(futures), desc=f"Processing frames in P{position}") as pbar:
            for future in as_completed(futures):
                results = future.result()
                position_results.extend(results)
                pbar.update(1)

    return position_results


def create_nuclear_manifest(dataset, num_workers=32):
    positions = []
    frames = []
    fov_paths = []
    nuclear_labels = []
    x_coords = []
    y_coords = []
    areas = []

    n_positions = dataset_io.get_total_number_of_positions(dataset)

    for position in range(n_positions):
        position_results = process_position(position, dataset, num_workers)
        for result in position_results:
            pos, frame, fov_path, label, x, y, area = result
            positions.append(pos)
            frames.append(frame)
            fov_paths.append(fov_path)
            nuclear_labels.append(label)
            x_coords.append(x)
            y_coords.append(y)
            areas.append(area)

    # Create a DataFrame from the lists
    df = pd.DataFrame(
        {
            "dataset": dataset,
            "position": positions,
            "frame": frames,
            "fov_path": fov_paths,
            "nuclear_label": nuclear_labels,
            "x": x_coords,
            "y": y_coords,
            "area": areas,
        }
    )

    out_put_path = set_output.get_output_path("nuclear_seg_manifests")
    df.to_parquet(f"{out_put_path}{dataset}_nuclear_manifest.parquet")

    return df


# %%
def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate nuclear manifest for a dataset.")
    parser.add_argument("dataset", type=str, help="The name of the dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    create_nuclear_manifest(args.dataset)
# %%
