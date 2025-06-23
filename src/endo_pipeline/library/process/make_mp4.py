from collections.abc import Callable
from pathlib import Path

import bioio_imageio
import fire
import numpy as np
from bioio import BioImage
from bioio.writers.timeseries_writer import TimeseriesWriter
from tqdm import tqdm


def make_mp4(
    img_path: str,
    out_path: str | None = None,
    fps: int = 20,
    sorting_function: Callable | None = None,
) -> None:
    """
    Takes a filepath to an image or a directory
    of images and saves them as a single .mp4 movie
    using BioIO's TimeSeriesWriter.

    Parameters
    ----------
    img_path: str
        a path to a file or directory of the images
        one wishes to turn in to a movie
    out_dir: str or None
        the path (incl. filename) where the movie should
        be saved
        if None, saves to the parent folder of img_path
        with the
    fps: int
        number of frames-per-second to save the mp4 with
    sorting_function: callable | None
        if provided, will sort the files found in
        img_path according to the provided function
        using the filenames are arguments

    Returns
    -------
    None
        the .mp4 movie is saved to out_dir
    """

    # convert the user-input paths to Path objects
    img_path_as_path = Path(img_path)
    out_path_as_path = Path(out_path) if out_path else img_path_as_path.parent

    # if the provided path is a folder then iterate
    # through the files in that folder (non-recursively)
    if img_path_as_path.is_dir():
        img_path_list = list(img_path_as_path.glob("*"))
        img_path_list = [fp for fp in img_path_list if fp.is_file()]  # only keep files
        if sorting_function:
            img_path_list = sorted(
                img_path_list, key=lambda fp: sorting_function(fp.stem)
            )
    else:
        img_path_list = [img_path_as_path]

    # open the provided image paths
    img_list = []
    for fp in tqdm(img_path_list, total=len(img_path_list), desc="Loading images"):
        # if the file is a .png or .gif then use the
        # bioio_imageio.Reader, otherwise let BioImage decide
        if fp.suffix.lower() == ".png":
            dim_order = "TYXS"
            reader = bioio_imageio.Reader
        else:
            dim_order = "TYX"
            reader = None
        img_list.append(BioImage(fp, reader=reader).get_image_data(dim_order).squeeze())

    # create an image array out of the
    img_arr = np.stack(img_list, axis=0)

    # if an existing directory was provided then
    # save the file there with the same filename
    # as was provided for the img_path
    if out_path_as_path.exists() and out_path_as_path.is_dir():
        out_filepath = out_path_as_path / f"{img_path_as_path.stem}.mp4"
    # if the provided out_path does not exist then
    # assume that this is a file path (not a directory)
    # and add the .mp4 file extension if no .mp4
    # extension was found
    else:
        if out_path_as_path.suffix != ".mp4":
            out_filepath = out_path_as_path.with_suffix(".mp4")
        else:
            out_filepath = out_path_as_path

    # if out_filepath already exists then add a number
    # until it is unique
    i = 1
    while out_filepath.exists():
        out_filepath = out_filepath.parent / f"{out_filepath.stem}_({i}).mp4"
        i += 1

    # save the .mp4 movie
    TimeseriesWriter.save(
        data=img_arr,
        uri=out_filepath,
        fps=fps,
    )


if __name__ == "__main__":
    fire.Fire(make_mp4)
