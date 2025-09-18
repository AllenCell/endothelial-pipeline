from typing import Literal

TAGS = ["preprocessing"]


def main(
    dataset_pair_type: Literal["live_fixed", "20X_40X"] = "live_fixed",
    resolution_level: int = 1,
    output_dir: str | None = None,
) -> None:
    """
    Register images from paired datasets and save the aligned images as multi-channel TIFF files.

    Default output directory is

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20X/40X.
    resolution_level
        The resolution level of the zarr files to be used for registration.
    output_dir
        The directory where the aligned images will be saved. If None, a default
        directory will be used.
    """
    import logging
    from pathlib import Path

    import tqdm

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.library.process.registration import (
        align_and_save_paired_images,
        concat_and_save_aligned_image_pairs,
    )
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    if output_dir is None:
        from endo_pipeline.settings import IF_INTEGRATION_SAVE_DIRECTORY

        output_path = Path(IF_INTEGRATION_SAVE_DIRECTORY)
    else:
        output_path = Path(output_dir)

    save_path = output_path / f"{dataset_pair_type}_resolution_{resolution_level}"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory set to: [ %s ]", output_path.as_posix())

    # When running workflow in demo mode, only the first two pairs of images
    # from the first dataset pair will be aligned and saved.
    if DEMO_MODE:
        num_datasets_to_align = 1
        num_positions_to_align = 4
        logger.warning(
            "Running in demo mode: only registering the first [ %s ] "
            "positions of the first [ %s ] dataset pair(s).",
            num_positions_to_align,
            num_datasets_to_align,
        )
    else:
        num_datasets_to_align = None
        num_positions_to_align = None

    # align the images and save the aligned file individually
    df = align_and_save_paired_images(
        dataset_pair_type,
        resolution_level,
        z_slice_offsets=Z_SLICE_OFFSETS,
        save_path=save_path,
        num_datasets_to_align=num_datasets_to_align,
        num_positions_to_align=num_positions_to_align,
    )

    # concatenate the aligned images and save them as multi-channel tiff files
    for row in tqdm.tqdm(df.itertuples()):
        row_dict = row._asdict()  # type: ignore[operator]
        out_path = concat_and_save_aligned_image_pairs(row_dict, save_path)
        logger.debug("Saved aligned image to [ %s ]", out_path.as_posix())


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
