from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path
from endo_pipeline.library.process.lib_grid_seg import (
    check_crop_indices_against_existing_segmentations,
    create_grid_segmentation_images,
    load_grid_diffae_df_for_tfe,
)


def main(datasets: Datasets | None, n_cores=4):
    """Creates grid-based segmentations based on the crop locations from the grid-based
    DiffAE dataframe of the first dataset in `datasets`, then checks that the crop indices
    subsequent datasets in `datasets` match the existing segmentations.
    """
    if datasets is None:
        datasets = get_datasets_in_collection("diffae_model_training")
        datasets.append(get_datasets_in_collection("replicate_2_datasets"))

    for i, dataset_name in enumerate(datasets):
        out_dir = get_output_path(__file__)
        out_dir.mkdir(parents=True, exist_ok=True)

        grid_df = load_grid_diffae_df_for_tfe(dataset_name)

        if i == 0:
            create_grid_segmentation_images(grid_df, out_dir)

        else:
            nm, df = zip(*grid_df.groupby(["position", "frame_number"]), strict=True)
            num_seg_files = len(nm)
            with ProcessPoolExecutor(max_workers=n_cores) as worker_pool:
                list(
                    tqdm(
                        worker_pool.map(
                            check_crop_indices_against_existing_segmentations,
                            df,
                            [out_dir] * num_seg_files,
                        ),
                        desc=f"Checking grid segmentations for {dataset_name}",
                        total=num_seg_files,
                    )
                )

    print("\N{PARTY POPPER} Done.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
