from typing import Literal

TAGS = ["diffae_model_finetuning"]


def main(
    dataset_pair_type: Literal["live_fixed", "20x_40x"] = "live_fixed",
    split: bool = True,
) -> None:
    """
    Generate a dataset of paired, aligned, brightfield images for finetuning a DiffAE model.

    Parameters
    ----------
    dataset_pair_type
        Whether paired datasets are live/fixed or 20x/40x.
    split
        Whether to split the dataset into training and validation sets (else, makes
        a single dataset CSV file that must be split manually).

    Returns
    -------
    :
        Saves the dataset as a CSV file in the output directory.
        The images will be saved as multi-channel TIFF files in the same directory.
    """
    import pandas as pd
    import tqdm
    from sklearn.model_selection import train_test_split

    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.process.registration import (
        align_and_save_paired_images,
        concat_and_save_aligned_image_pairs,
    )

    save_path = get_output_path(
        "finetune_paired_dataset", dataset_pair_type, include_timestamp=False
    )

    df = align_and_save_paired_images(dataset_pair_type, save_path)

    out_paths = [
        concat_and_save_aligned_image_pairs(row, save_path) for row in tqdm.tqdm(df.itertuples())
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
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
