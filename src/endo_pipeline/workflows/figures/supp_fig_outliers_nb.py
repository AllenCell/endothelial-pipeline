"""
Visualize the selection of Z slices for image preprocessing.
Visualize the automatic detection of outlier timepoints in BF and EGFP channels.

#supfig #preprocessing
"""

# %%
import matplotlib.pyplot as plt

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.process.timepoint_outliers import (
    plot_single_timepoint_bf_outliers,
    plot_single_timepoint_gfp_outliers,
    print_timepoint_annotation_performance_stats,
)
from endo_pipeline.library.process.z_stack_selection import (
    plot_global_center_plane,
    plot_histogram_upper_slices_available,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dataset_annotations import (
    IN_FOCUS_PLANE_MANIFEST_NAME,
    REPRESENTATIVE_ANNOTATION_POSITION,
    REPRESENTATIVE_ANNOTATION_TIMEPOINT,
    TIMEPOINT_OUTLIERS_MANIFEST_NAME,
)
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import LOWER_Z_SLICE_OFFSET, UPPER_Z_SLICE_OFFSET

plt.style.use("endo_pipeline.figure")

# %% Load dataset
FIGURE_ID = "SUPP_FIG_Z_SLICE"
dataset = EXAMPLE_DATASET[FIGURE_ID]
save_dir_1 = get_output_path("supp_fig_z_slice_selection")
dataset_config = load_dataset_config(dataset)
position = REPRESENTATIVE_ANNOTATION_POSITION
timepoint = REPRESENTATIVE_ANNOTATION_TIMEPOINT

# %% Load dataframe for in focus plane annotations
in_focus_plane_df_manifest = load_dataframe_manifest(IN_FOCUS_PLANE_MANIFEST_NAME)
in_focus_plane_df_location = in_focus_plane_df_manifest.locations[dataset]
in_focus_plane_df = load_dataframe(in_focus_plane_df_location).set_index(Column.POSITION)
in_focus_plane = in_focus_plane_df.loc[position].to_dict()

# %% Panel - In focus Z slice selection per timepoint
stdevs = in_focus_plane[Column.Annotations.CENTER_PLANE_SLICES_STD_DEVS]
center_plane = in_focus_plane[Column.Annotations.CENTER_PLANE_MEAN]
plot_standard_devs_per_slice(
    stdevs,
    center_plane,
    dataset,
    position,
    timepoint,
    save_dir_1,
    (2.4, 2.15),
)

# %% Panel - In focus Z slice selection per position over time
focal_planes = in_focus_plane[Column.Annotations.CENTER_PLANES]
plot_global_center_plane(
    focal_planes,
    dataset_config.name,
    position,
    save_dir_1,
    (2.4, 2.15),
    show_histogram=False,
)

# %% Panel - Distribution of upper slices available across datasets
datasets = get_datasets_in_collection("shear_stress")
plot_histogram_upper_slices_available(datasets, save_dir_1, figure_size=(1.5, 2.15))

# %% Panel - Example images of selected Z slices
visualize_slice_selection(
    dataset_config,
    center_plane,
    position,
    timepoint,
    save_dir_1,
    (MAX_FIGURE_WIDTH * 0.7, 3),
    LOWER_Z_SLICE_OFFSET,
    UPPER_Z_SLICE_OFFSET,
)

# %% Load example datasets
FIGURE_ID = "SUPP_FIG_SINGLE_TP"
save_dir_2 = get_output_path("annotate_tp_outliers")
dataset_bf = EXAMPLE_DATASET[f"{FIGURE_ID}_BF_OUTLIER"]
dataset_gfp = EXAMPLE_DATASET[f"{FIGURE_ID}_GFP_OUTLIER"]

# %% Load dataframe for timepoint outlier annotations
tp_outliers_df_manifest = load_dataframe_manifest(TIMEPOINT_OUTLIERS_MANIFEST_NAME)
bf_tp_outliers_df_location = tp_outliers_df_manifest.locations[dataset_bf]
bf_tp_outliers_df = load_dataframe(bf_tp_outliers_df_location).set_index(Column.POSITION)
bf_tp_outliers = bf_tp_outliers_df.loc[position].to_dict()
gfp_tp_outliers_df_location = tp_outliers_df_manifest.locations[dataset_gfp]
gfp_tp_outliers_df = load_dataframe(gfp_tp_outliers_df_location).set_index(Column.POSITION)
gfp_tp_outliers = gfp_tp_outliers_df.loc[position].to_dict()

# %% Panel - Auto-detect BF outliers
plot_single_timepoint_bf_outliers(
    mean_intensity=bf_tp_outliers[Column.Annotations.BF_MEAN_INTENSITY],
    rolling_median=bf_tp_outliers[Column.Annotations.BF_ROLLING_MEDIAN],
    dark_threshold=bf_tp_outliers[Column.Annotations.BF_PARTIAL_DARK_THRESHOLD],
    bright_threshold=bf_tp_outliers[Column.Annotations.BF_BRIGHT_THRESHOLD],
    dark_outliers=sorted(
        set(
            bf_tp_outliers[Column.Annotations.BF_DARK_OUTLIERS].astype(int).tolist()
            + bf_tp_outliers[Column.Annotations.BF_PARTIAL_DARK_OUTLIERS].astype(int).tolist()
        )
    ),
    bright_outliers=bf_tp_outliers[Column.Annotations.BF_BRIGHT_OUTLIERS].astype(int),
    dataset_name=dataset_bf,
    position=position,
    save_dir=save_dir_2,
    figure_size=(3.4, 2.5),
)

# %% Panel - Auto-detect EGFP scope errors
plot_single_timepoint_gfp_outliers(
    timepoint_means=gfp_tp_outliers[Column.Annotations.GFP_TIMEPOINT_MEANS],
    rolling_median=gfp_tp_outliers[Column.Annotations.GFP_ROLLING_MEDIAN],
    lower_threshold=gfp_tp_outliers[Column.Annotations.GFP_LOWER_THRESHOLD],
    upper_threshold=gfp_tp_outliers[Column.Annotations.GFP_UPPER_THRESHOLD],
    dark_outliers=gfp_tp_outliers[Column.Annotations.GFP_DARK_OUTLIERS],
    bright_outliers=gfp_tp_outliers[Column.Annotations.GFP_BRIGHT_OUTLIERS],
    dataset_name=dataset_gfp,
    position=position,
    save_dir=save_dir_2,
    figure_size=(3.2, 2.5),
)

# %% Performance statistics reported across datasets in collection
datasets = get_datasets_in_collection("shear_stress", "perturbation")

bf_results = print_timepoint_annotation_performance_stats(
    datasets=datasets,
    manual_annotations=[
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
    ],
    auto_annotations=[
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
    ],
    annotation_type="brightfield_outlier_detection",
)
print(bf_results)

gfp_results = print_timepoint_annotation_performance_stats(
    datasets=datasets,
    manual_annotations=[TimepointAnnotation.GFP_SCOPE_ERROR],
    auto_annotations=[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR],
    annotation_type="gfp_outlier_detection",
)
print(gfp_results)

# %% Figure
panels = [
    FigurePanel(
        letter="A",
        path=save_dir_2 / f"bf_outliers_{dataset_bf}_P{position}.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=save_dir_2 / f"gfp_outliers_{dataset_gfp}_P{position}.svg",
        x_position=3.5,
        y_position=0,
        x_offset=-0.1,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="C",
        path=save_dir_1 / f"standard_devs_{dataset}_P{position}_{timepoint}.svg",
        x_position=0,
        y_position=2.6,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="D",
        path=save_dir_1 / f"global_center_plane_{dataset}_P{position}.svg",
        x_position=2.5,
        y_position=2.6,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="E",
        path=save_dir_1 / "n_slices_above_in_focus_z_histogram.svg",
        x_position=4.85,
        y_position=2.6,
        x_offset=0.08,
        y_offset=0.2,
    ),
    FigurePanel(
        letter="F",
        path=save_dir_1
        / f"plane_selection_vis_{dataset}_P{position}_{timepoint}_offset{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}_scalebar100um.svg",
        x_position=0,
        y_position=2.3 + 2.6,
        x_offset=0.08,
        y_offset=0.08,
    ),
]

output_path = (
    get_output_path("supp_fig_z_slice_outliers") / "supp_fig_z_slice_selection_outliers.svg"
)
build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT)

# %%
