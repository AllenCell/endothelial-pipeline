# %%
import fire

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.visualize.crop_montage import create_montage, specify_crops


def main(
    dataset_names: str | list[str] | None = None,
    pc_axis: int = 1,
    pc_val: float = 0.25,
    frame_range: list[int] | None = None,
    plot_heatmap: bool = False,
) -> None:

    fig_savedir = get_output_path("crop_visualization")

    df, pca, model_manifest_list = specify_crops.load_data(dataset_names)

    df_filtered = specify_crops.filter_dataframe(
        df,
        pc_axis,
        pc_val,
        model_manifest_list,
        pca,
        fig_savedir,
        frame_range,
        plot_heatmap,
    )

    df_sample = specify_crops.sample_dataframe(df_filtered)

    create_montage.generate_contact_sheet(
        df_sample,
        pc_axis,
        pc_val,
        fig_savedir,
    )


if __name__ == "__main__":
    fire.Fire(main)

# %%
