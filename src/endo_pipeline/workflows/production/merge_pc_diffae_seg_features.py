from endo_pipeline.cli import Datasets


def main(datasets: Datasets, n_proc: int = 1) -> None:
    """Generates merged dataframes of PCA-reduced DiffAE features and live segmentation features."""
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    from endo_pipeline.library.analyze.integration.track_integration import (
        get_and_save_pc_diffae_feats_liveseg_feats_merged_table,
    )

    with ProcessPoolExecutor(n_proc) as executor:
        list(
            tqdm(
                executor.map(get_and_save_pc_diffae_feats_liveseg_feats_merged_table, datasets),
                total=len(datasets),
            )
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
