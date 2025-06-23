from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from bioio import BioImage
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    concatenate_and_save_feature_tables,
    extract_T,
    fire_parse_generate_dataset_name_list,
    get_cdh5_classic_segmentation_path,
    get_dataset_info,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
    load_dataset_position_as_dask_array,
    save_git_versioning_info,
)
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze import shape_features as feat
from src.endo_pipeline.library.process.general_image_preprocessing import (
    build_analysis_queue,
    save_image_output,
)


def build_measured_features_tables_multiproc_wrapper(args: dict) -> None:
    dataset_name = args["dataset_name"]
    scene = args["scene_index"]
    position = args["position"]
    T = args["T"]
    img_bin_level = args["image_bin_level"]
    save_output = args["save_output"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]
    use_sldy_data = args["use_sldy_data"]
    create_validation_image = args["validation_image"]
    build_measured_features_tables(
        dataset_name,
        T,
        out_dir,
        scene,
        position,
        use_sldy_data,
        img_bin_level,
        save_output=save_output,
        create_validation_image=create_validation_image,
        verbose=verbose,
    )


def build_measured_features_tables(
    dataset_name: str,
    T: int,
    out_dir: str | Path,
    scene: str | int = 0,
    position: int = 0,
    use_sldy_data: bool | None = False,
    img_bin_level: int = 0,
    save_output: bool | None = True,
    create_validation_image: bool = False,
    verbose: bool = True,
) -> None:
    """
    Builds tables of measured features from the segmentation images
    and the raw cdh5 images.
    The segmentation properties tables is
    a table of measured features extracted from the cdh5 segmentations
    using skimage.regionprops.
    The edge alignments table contains measured features that were
    determined based on a thresholded image of the cdh5 signal
    (i.e. they don't require the cdh5 segmentations).

    Also produces a validation image if requested
    (a validation image has segmentation borders, nodes, edges, and
    the straight lines connecting nodes as channels in a single
    .tiff image).

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to process.
    T: int
        The timepoint to process.
    out_dir: str | Path
        The output directory to save the tables and validation images to.
    scene: str | int
        The scene index to process (only used if use_sldy_data = True).
    position: int
        The position to process (this will be equal to the scene index).
    use_sldy_data: bool | None
        Whether to use the original data or the zarr data.
    img_bin_level: int
        The binning level to use when loading an image (only used if use_sldy_data = False).
        Currently not implemented.
    save_output: bool | None
        Whether to save the output tables (and validation images if selected).
    create_validation_image: bool
        Whether to create a validation image.
    verbose: bool
        Whether to print progress messages.

    Returns
    -------
    This function will only save tables and images,
    it does not return anything.

    The tables contain the following information:
    segmentation properties table:
    - filepath_raw_image
    - filepath_segmentation_image
    - dataset_name
    - scene_index
    - position
    - T
    - cell_label
    - cell_centroid
    - cell_area (px**2)
    - cell_perimeter (px)
    - cell_perimeter (px)
    - cell_solidity
    - major_axis_length
    - minor_axis_length
    - cell_eccentricity
    - cell_orientation
    - cell_fluorescence_mean (a.u.)
    - cell_fluorescence_std (a.u.)
    - cell_fluorescence_median (a.u.)
    - cell_fluoresnce_min (a.u.)
    - cell_fluorescence_pct25 (a.u.)
    - cell_fluorescence_pct75 (a.u.)
    - cell_fluorescence_max (a.u.)
    - neighboring_cell_labels
    - edge_labels
    - node_labels
    - node_pair_labels
    - touches_image_border
    - measurement_timestamp
    - git_branch_name
    - git_commit_hash
    - git_uncommitted_changes

    edge alignments table:
    - filepath_raw_image
    - filepath_raw_image
    - filepath_segmentation_image
    - dataset_name
    - scene_index
    - position
    - T
    - node_pair_labels
    - node_pair_centroids
    - node_to_node_distance
    - angle_relative_to_horizontal
    - connecting_edges
    - edge_num_pixels
    - edge_length (px)
    - edge_fluorescence_mean (a.u.)
    - edge_fluorescence_std (a.u.)
    - edge_fluorescence_median (a.u.)
    - edge_fluoresnce_min (a.u.)
    - edge_fluorescence_pct25 (a.u.)
    - edge_fluorescence_pct75 (a.u.)
    - edge_fluorescence_max (a.u.)
    - measurement_timestamp
    - git_branch_name
    - git_commit_hash
    - git_uncommitted_changes
    """

    print(f"Working on {dataset_name} -- T={T}...") if verbose else None

    dim_order = "TCZYX"

    out_dir = Path(out_dir)
    images_out_dir = out_dir / f"{dataset_name}/P{position}/images"
    tables_out_dir_alignments = out_dir / f"{dataset_name}/P{position}/tables_alignments"
    tables_out_dir_segprops = out_dir / f"{dataset_name}/P{position}/tables_segmentation_properties"

    print(f"T={T} -- loading imaging datasets") if verbose else None
    # load the raw cdh5 image data
    if use_sldy_data:
        cdh5_chan_index = get_dataset_info(dataset_name)["channel_488_index"]
        image_path = Path(get_original_path(dataset_name))
        img = BioImage(image_path)
        img.set_scene(scene)
        raw_dask_arr = img.get_image_dask_data(dim_order, C=[cdh5_chan_index], T=T)
        raw_dask_arr = raw_dask_arr.max(axis=dim_order.index("Z"), keepdims=True)
        raw_arr = raw_dask_arr.compute().squeeze()
        voxel_size = img.physical_pixel_sizes
    else:
        raw_arr = load_dataset_position_as_dask_array(
            dataset_name,
            position,
            channels=["EGFP"],
            time_start=T,
            time_end=T,
        )
        raw_arr = raw_arr.max(axis=dim_order.index("Z")).squeeze().compute()
        zarr_name = get_zarr_name(dataset_name, position)
        image_path = Path(get_zarr_path(dataset_name)[zarr_name])
        voxel_size = BioImage(image_path).physical_pixel_sizes

    print(f"T={T} -- loading classic segmentation") if verbose else None
    # load the segmentation images
    seg_dir = get_cdh5_classic_segmentation_path(dataset_name, position)
    if seg_dir is not None:
        seg_dir = Path(seg_dir)
    else:
        print(f"No segmentation directory found for {dataset_name}. Skipping cdh5 measurements.")
        return

    seg_filepath_list = [fp for fp in seg_dir.glob("*.ome.tif*") if extract_T(fp.name) == T]
    assert (
        len(seg_filepath_list) == 1
    ), f"Found {len(seg_filepath_list)} segmentation files for T={T} in {dataset_name}. Expected 1."
    seg_filepath = seg_filepath_list[0]
    seg = BioImage(seg_filepath)
    seg_arr = seg.get_image_dask_data(dim_order).compute().squeeze()
    # NOTE: the segmentation images are stored as a single channel and single timepoint
    seg_borders = find_boundaries(seg_arr)

    ## convert cleaned up threshold of cadherin signal to nodes and edges
    print(f"T={T} -- getting nodes and edges") if verbose else None
    nodes, edges, skel, conn = feat.arr2graph(seg_borders, closing_step=False)

    ## get the node-to-node distances and the angle between a line connecting two nodes
    ## and a horizontal line
    ## NOTE there should also be a way to get the error in the measurement of the angles too...
    (
        print(f"T={T} -- calculating distances and angles between neighboring nodes")
        if verbose
        else None
    )
    neighbor_node_metrics, labeled_region_metrics = feat.calculate_region_border_metrics(
        seg_borders.astype(bool), raw_arr, seg_arr, verbose=verbose
    )

    ## save a table of the results
    if save_output:
        tables_out_dir_alignments.mkdir(exist_ok=True, parents=True)
        ## save table output of edge alignments
        (print(f"T={T} -- saving table of edge angles and distances") if verbose else None)
        table = pd.DataFrame(
            {
                "filepath_raw_image": image_path,
                "filepath_segmentation_image": seg_filepath,
                "dataset_name": dataset_name,
                "scene_index": scene,
                "position": position,
                "T": T,
                "node_pair_labels": neighbor_node_metrics["node_pair_labels"],
                "node_pair_centroids": neighbor_node_metrics["node_pair_centroids"],
                "node_to_node_distance": neighbor_node_metrics["distances"],
                "angle_relative_to_horizontal": neighbor_node_metrics["angles"],
                "connecting_edges": neighbor_node_metrics["edge_labels"],
                "edge_num_pixels": neighbor_node_metrics["edge_num_pixels"],
                "edge_length (px)": neighbor_node_metrics["length (px)"],
                "edge_fluorescence_mean (a.u.)": neighbor_node_metrics["fluor_mean (au)"],
                "edge_fluorescence_std (a.u.)": neighbor_node_metrics["fluor_std (au)"],
                "edge_fluorescence_median (a.u.)": neighbor_node_metrics["fluor_median (au)"],
                "edge_fluoresnce_min (a.u.)": neighbor_node_metrics["fluor_min (au)"],
                "edge_fluorescence_pct25 (a.u.)": neighbor_node_metrics["fluor_pct25 (au)"],
                "edge_fluorescence_pct75 (a.u.)": neighbor_node_metrics["fluor_pct75 (au)"],
                "edge_fluorescence_max (a.u.)": neighbor_node_metrics["fluor_max (au)"],
            }
        )
        table.to_csv(
            tables_out_dir_alignments / f"{dataset_name}_P{position}_T{T}_alignments.tsv",
            index=False,
            sep="\t",
        )

        if create_validation_image:
            images_out_dir.mkdir(exist_ok=True, parents=True)
            ## save images containing the nodes, edges, and node-node lines
            ## as different channels
            (
                print(f"T={T} -- saving multichannel images of results for validation")
                if verbose
                else None
            )
            ## create a rasterized image of the lines
            lines = np.zeros(nodes.shape, dtype=np.uint16)
            ## need to flatten the node_coord_pairs first before passing to rasterize_edge_between_nodes
            node_coord_pairs = [
                node_coords
                for edge in neighbor_node_metrics["node_pair_centroids"]
                for node_coords in edge
            ]
            lines, line_labels_dict = feat.rasterize_edges_between_nodes(
                node_coord_pairs, lines, label_lines=True
            )

            ## organize the image data and save it
            out_path = images_out_dir / f"{dataset_name}_P{position}_T{T}.ome.tiff"
            images_out = [seg_borders, nodes, edges, lines]
            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": ["segmentation_borders", "nodes", "edges", "lines"],
                "channel_colors": [
                    (255, 255, 255),
                    (255, 0, 255),
                    (0, 255, 255),
                    (255, 255, 0),
                ],
                "physical_pixel_sizes": voxel_size,
                "dim_order": "YX",
            }
            save_image_output(out_path, images_out, images_out_metadata)

        ## save table output of cell properties (e.g. areas, etc.)
        if labeled_region_metrics:
            tables_out_dir_segprops.mkdir(exist_ok=True, parents=True)
            print(f"T={T} -- saving table of cell properties") if verbose else None
            table = pd.DataFrame(
                {
                    "filepath_raw_image": image_path,
                    "filepath_segmentation_image": seg_filepath,
                    "dataset_name": dataset_name,
                    "scene_index": scene,
                    "position": position,
                    "T": T,
                    "cell_label": labeled_region_metrics["cell_label"],
                    "cell_centroid": labeled_region_metrics["cell_centroid"],
                    "cell_area (px**2)": labeled_region_metrics["cell_area (px**2)"],
                    "cell_perimeter (px)": labeled_region_metrics["cell_perimeter (px)"],
                    "cell_solidity": labeled_region_metrics["cell_solidity"],
                    "major_axis_length": labeled_region_metrics["major_axis_length"],
                    "minor_axis_length": labeled_region_metrics["minor_axis_length"],
                    "cell_eccentricity": labeled_region_metrics["cell_eccentricity"],
                    "cell_orientation": labeled_region_metrics["cell_orientation"],
                    "cell_fluorescence_mean (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_mean (au)"
                    ],
                    "cell_fluorescence_std (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_std (au)"
                    ],
                    "cell_fluorescence_median (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_median (au)"
                    ],
                    "cell_fluoresnce_min (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_min (au)"
                    ],
                    "cell_fluorescence_pct25 (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_pct25 (au)"
                    ],
                    "cell_fluorescence_pct75 (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_pct75 (au)"
                    ],
                    "cell_fluorescence_max (a.u.)": labeled_region_metrics[
                        "cell_fluorescence_max (au)"
                    ],
                    "neighboring_cell_labels": labeled_region_metrics["neighboring_cell_labels"],
                    "edge_labels": labeled_region_metrics["edge_labels"],
                    "node_labels": labeled_region_metrics["node_labels"],
                    "node_pair_labels": labeled_region_metrics["node_pair_labels"],
                    "touches_image_border": labeled_region_metrics["touches_image_border"],
                }
            )
            table.to_csv(
                tables_out_dir_segprops / f"{dataset_name}_P{position}_T{T}_segprops.tsv",
                index=False,
                sep="\t",
            )


def concatenate_tables(dataset_name: str, out_dir: str | Path) -> None:
    print(f"- {dataset_name}")
    # get the alignment table paths and segmentation properties
    # table paths for each dataset
    out_dir = Path(out_dir)
    tables_alignments = out_dir.glob(f"**/{dataset_name}/*/tables_alignments/*.csv")
    tables_segprops = out_dir.glob(f"**/{dataset_name}/*/tables_segmentation_properties/*.csv")

    # concatenate and save the tables for each dataset
    concatenated_table_out_dir = out_dir / f"{dataset_name}"

    master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_alignments])
    master_table.to_csv(concatenated_table_out_dir / f"{dataset_name}_alignments.csv", index=False)

    master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_segprops])
    master_table.to_csv(
        concatenated_table_out_dir / f"{dataset_name}_segmentation_properties.csv",
        index=False,
    )


def concatenate_tables_multiproc(queue_group: tuple) -> None:
    dataset_name, queue_df = queue_group
    out_dir = queue_df["output_dir"].iloc[0]
    concatenate_tables(dataset_name, out_dir)


def main(
    n_proc: int = 1,
    dataset_name: str | None = None,
    save_output: bool = True,
    is_test: bool = False,
    verbose: bool = False,
    use_sldy_data: bool = False,
) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    print("Building analysis queue...")
    out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=None,
        use_sldy_data=use_sldy_data,
    )

    if n_proc > 1:
        if __name__ == "__main__":
            with Pool(processes=n_proc) as pool:
                list(
                    tqdm(
                        pool.imap(
                            build_measured_features_tables_multiproc_wrapper,
                            analysis_queue,
                            chunksize=2,
                        ),
                        total=len(analysis_queue),
                        desc="Getting cell features (MP)...",
                    )
                )
                pool.close()
                pool.join()
    else:
        for dataset_name_and_args in tqdm(analysis_queue, desc="Getting cell features (1P)..."):
            build_measured_features_tables_multiproc_wrapper(dataset_name_and_args)

    # lastly, for each dataset concatenate the tables from each timepoint
    # into a single output table for that dataset
    if save_output:
        for dataset_name in dataset_name_list:
            concatenate_and_save_feature_tables(
                out_dir,
                dataset_name,
                out_file_suffix="alignments",
                input_filename_contains="alignments",
                file_extension=".tsv",
                remove_initial_files_and_folders=True,
            )
            concatenate_and_save_feature_tables(
                out_dir,
                dataset_name,
                out_file_suffix="segprops",
                input_filename_contains="segprops",
                file_extension=".tsv",
                remove_initial_files_and_folders=True,
            )

        # save git versioning info
        save_git_versioning_info(
            out_dir=out_dir, filename_prefix=f"{Path(__file__).stem}", verbose=verbose
        )

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
