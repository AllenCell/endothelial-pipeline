from pathlib import Path

import fire
import numpy as np
import pandas as pd
import tqdm
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from sklearn.model_selection import train_test_split

from cellsmap.util.set_output import get_output_path


def find_csvs(base_path: Path, directories: list[str] | None = None):
    if directories is None:
        directories = [p for p in base_path.glob("*_vs_*") if p.is_dir()]
        for d in directories:
            files = list(d.glob("aligned*.csv"))
            if len(files) != 1:
                directories.remove(d)

    csv_paths = [list((base_path / d).glob("aligned_*.csv"))[0] for d in directories]

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df = df.dropna(subset=["fixed", "moving"])
    return df


def _get_concat_path(row, savedir):
    """
    Generate a path for the concatenated image based on the fixed image path.
    The moving image path is not used in the final file name.
    """
    return savedir / f"{str(Path(row.fixed).stem).replace('_fixed', '')}.ome.tiff"


def concat(row, savedir):
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


def generate_paired_dataset(
    base_path: str, directories: list[str] | None = None, split: bool = False
):
    save_path = Path(get_output_path("finetune_paired_dataset"))
    base_path = Path(base_path)

    df = find_csvs(base_path, directories)
    print(f"Found {len(df)} pairs of images to save")

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
    fire.Fire(generate_paired_dataset)
