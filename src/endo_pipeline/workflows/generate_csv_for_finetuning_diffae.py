import logging
from pathlib import Path
from typing import Literal

import fire
import numpy as np
import pandas as pd
import tqdm
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from sklearn.model_selection import train_test_split

from src.endo_pipeline.configs import get_datasets_in_collection
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.process.registration import align_all_positions

logger = logging.getLogger(__name__)


def _get_concat_path(row: pd.Series, savedir: Path) -> Path:
    """
    Generate a path for the concatenated image based on the fixed image path.
    The moving image path is not used in the final file name.
    """
    return savedir / f"{str(Path(row.fixed).stem).replace('_fixed', '')}.ome.tiff"


def _get_paired_dataset_dict(
    dataset_pair_type: Literal["live_fixed", "20X_40X"],
) -> dict[str, list[str]]:

    # Get the list of datasets of the specified pair type.
    dataset_list = get_datasets_in_collection(f"{dataset_pair_type}_paired_datasets")

    # Set dataset name flags for setting
    # "fixed" and "moving" images for alignment.
    if dataset_pair_type == "live_fixed":
        # for live/fixed pairs, the "fixed" image
        # for alignment is the pre-fixation (live) image
        # and the "moving" image is the post-fixation (fixed) image.
        fixed_flag = "PreFixation"
        moving_flag = "PostFixation"
    else:
        # for 20x/40x pairs, the "fixed" image is the 20x image
        # and the "moving" image is the 40x image.
        fixed_flag = "20X"
        moving_flag = "40X"
    dataset_pairs = {
        "fixed": [dataset_name for dataset_name in dataset_list if fixed_flag in dataset_name],
        "moving": [dataset_name for dataset_name in dataset_list if moving_flag in dataset_name],
    }
    if len(dataset_pairs["fixed"]) != len(dataset_pairs["moving"]):
        logger.error("Mismatch in number of fixed and moving datasets for image alignment.")
        raise ValueError(
            f"Found {len(dataset_pairs['fixed'])} fixed datasets and "
            f"{len(dataset_pairs['moving'])} moving datasets. "
            "Please check the dataset names in the collection."
        )
    return dataset_pairs


def _align_and_save_paired_images(
    dataset_pair_type: Literal["live_fixed", "20x_40x"],
    save_path: Path,
) -> pd.DataFrame:
    # Note that the "fixed" key refers to the image being used as
    # the reference image for alignment, and the "moving" key
    # refers to the image being aligned to the fixed image.
    # That is, "fixed" here does not refer to the image being fixed.

    dataset_pairs = _get_paired_dataset_dict(dataset_pair_type)

    fixed_datasets = dataset_pairs["fixed"]
    moving_datasets = dataset_pairs["moving"]

    alignment_method = "sift" if dataset_pair_type == "live_fixed" else "template"

    df = []
    for fixed, moving in zip(fixed_datasets, moving_datasets):
        df.append(
            align_all_positions(
                fixed,
                moving,
                save_path,
                alignment_method=alignment_method,
            )
        )
    df = pd.concat(df, ignore_index=True)
    df = df.dropna(subset=["fixed", "moving"])
    print(f"Found {len(df)} pairs of images to save")
    return df


def _concat_and_save_aligned_image_pairs(row: pd.Series, savedir: Path) -> Path:
    save_path = _get_concat_path(row, savedir)
    if save_path.exists():
        print(f"Skipping {save_path} as it already exists.")
        return save_path
    # take standard deviation projection here to allow concatenation with different z-axis sizes
    fixed = BioImage(row.fixed).data.squeeze().max(0)
    moving = BioImage(row.moving).data.squeeze().max(0)

    out = np.stack([fixed, moving], axis=0)[:, None]

    OmeTiffWriter.save(uri=save_path, data=out)
    return save_path


def main(
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    split: bool = True,
):
    """
    Utility function for generating a dataset of paired, aligned,
    brightfield images for finetuning a DiffAE model.

    Parameters
    ----------
    dataset_pair_type: Literal['live_fixed', '20x_40x']
        Whether paired dataset is aligned live/fixed or 20x/40x. This will
        determine the directory structure to search for aligned image pairs.
    split: bool
        If True, the dataset will be split into training and validation sets.
        The split will be saved as `train.csv` and `val.csv`. If False, the
        entire dataset will be saved as `dataset.csv`, and no split will be performed.
        Note that in this case the split must be performed manually before training.
    """
    save_path = get_output_path(
        "finetune_paired_dataset", dataset_pair_type, include_timestamp=False
    )

    df = _align_and_save_paired_images(dataset_pair_type, save_path)

    out_paths = [
        _concat_and_save_aligned_image_pairs(row, save_path) for row in tqdm.tqdm(df.itertuples())
    ]

    out_df = pd.DataFrame({"path": out_paths, "channel": ["0,1"] * len(out_paths)})

    if split:
        train, test = train_test_split(out_df, test_size=0.2, random_state=42)
        train.to_csv(save_path / "train.csv", index=False)
        test.to_csv(save_path / "val.csv", index=False)
    else:
        out_df.to_csv(save_path / "dataset.csv", index=False)
    print(f"Saved dataset to {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
