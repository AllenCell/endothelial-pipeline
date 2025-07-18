from pathlib import Path
from typing import Literal

import fire
import numpy as np
import pandas as pd
import tqdm
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from sklearn.model_selection import train_test_split

from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.process.registration import align_all_positions


def _get_concat_path(row: pd.Series, savedir: Path) -> Path:
    """
    Generate a path for the concatenated image based on the fixed image path.
    The moving image path is not used in the final file name.
    """
    return savedir / f"{str(Path(row.fixed).stem).replace('_fixed', '')}.ome.tiff"


def _get_aligned_paths(
    dataset_type: Literal["live_fixed", "20x_40x"],
    save_path: Path,
    fixed_datasets: list[str] | None = None,
    moving_datasets: list[str] | None = None,
) -> pd.DataFrame:
    datasets = {
        "live_fixed": {
            "fixed": ["20250214_pairedPreFixation"],
            "moving": ["20250214_pairedPostFixation"],
        },
        "20x_40x": {
            "fixed": ["20250110_paired20X", "20250227_paired20X", "20250228_paired20X"],
            "moving": [
                "20250110_paired40X",
                "20250227_paired40X",
                "20250228_paired40X",
            ],
        },
    }[dataset_type]

    fixed_datasets = fixed_datasets or datasets["fixed"]
    moving_datasets = moving_datasets or datasets["moving"]

    df = []
    for fixed, moving in zip(fixed_datasets, moving_datasets):
        df.append(
            align_all_positions(
                fixed,
                moving,
                save_path,
                alignment_method="sift" if dataset_type == "live_fixed" else "template",
            )
        )
    df = pd.concat(df, ignore_index=True)
    df = df.dropna(subset=["fixed", "moving"])
    print(f"Found {len(df)} pairs of images to save")
    return df


def concat(row: pd.Series, savedir: Path) -> Path:
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
    dataset_type: Literal["live_fixed", "20x_40x"],
    fixed_datasets: list[str] | None = None,
    moving_datasets: list[str] | None = None,
    split: bool = True,
):
    """
    Utility function for generating a dataset of paired, aligned, brightfield images for finetuning a DiffAE model.

    Parameters
    ----------
    dataset_type: Literal['live_fixed', '20x_40x']
        Whether paired dataset is aligned live/fixed or 20x/40x. This will determine the directory structure to search for aligned image pairs. If `model_name` matches `fixed_finetuned_model_name`, then `dataset_type` should be `live_fixed`. If `model_name` matches `model_name` from `paired_data_validation`, then `dataset_type` should be `20x_40x`.
    fixed_datasets: list[str] | None
        A list of fixed datasets to use for generating the dataset. If None, the function will use the default datasets for the specified `dataset_type`.
    moving_datasets: list[str] | None
        A list of moving datasets to use for generating the dataset. The order should be paired with fixed_datasets. If None, the function will use the default datasets for the specified `dataset_type`.
    split: bool
        If True, the dataset will be split into training and validation sets. The split will be saved as `train.csv` and `val.csv`. If False, the entire dataset will be saved as `dataset.csv`.
    """
    save_path = get_output_path("finetune_paired_dataset", dataset_type, include_timestamp=False)

    df = _get_aligned_paths(dataset_type, save_path, fixed_datasets, moving_datasets)

    out_paths = [concat(row, save_path) for row in tqdm.tqdm(df.itertuples())]

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
