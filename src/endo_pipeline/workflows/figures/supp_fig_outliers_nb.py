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
from endo_pipeline.library.process.single_tp_outlier.bf_timepoint_outlier import detect_bf_outliers
from endo_pipeline.library.process.single_tp_outlier.gfp_timepoint_outlier import (
    detect_egfp_scope_errors,
)
from endo_pipeline.library.process.single_tp_outlier.timepoint_outlier import performance_stats
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
dataset_config_bf = load_dataset_config(EXAMPLE_DATASET[f"{FIGURE_ID}_BF_OUTLIER"])
dataset_config_gfp = load_dataset_config(EXAMPLE_DATASET[f"{FIGURE_ID}_GFP_OUTLIER"])
position = 0

# %% Panel - Auto-detect BF outliers
_ = detect_bf_outliers(dataset_config_bf, position, visualize=True, figure_size=(3.4, 2.5))

# %% Panel - Auto-detect EGFP scope errors
_ = detect_egfp_scope_errors(dataset_config_gfp, position, visualize=True, figure_size=(3.2, 2.5))

# %% Performance statistics reported across datasets in collection
datasets = get_datasets_in_collection("shear_stress") + get_datasets_in_collection("perturbation")

bf_results = performance_stats(
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

gfp_results = performance_stats(
    datasets=datasets,
    manual_annotations=[TimepointAnnotation.GFP_SCOPE_ERROR],
    auto_annotations=[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR],
    annotation_type="gfp_outlier_detection",
)
print(gfp_results)

# %% Figure
save_dir_2 = get_output_path("annotate_tp_outliers")
panels = [
    FigurePanel(
        letter="A",
        path=save_dir_1 / f"standard_devs_{dataset}_P{position}_{timepoint}.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="B",
        path=save_dir_1 / f"global_center_plane_{dataset}_P{position}.svg",
        x_position=2.5,
        y_position=0,
        x_offset=0,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="C",
        path=save_dir_1 / "n_slices_above_in_focus_z_histogram.svg",
        x_position=4.85,
        y_position=0,
        x_offset=0.08,
        y_offset=0.2,
    ),
    FigurePanel(
        letter="D",
        path=save_dir_1
        / f"plane_selection_vis_{dataset}_P{position}_{timepoint}_offset{LOWER_Z_SLICE_OFFSET}_{UPPER_Z_SLICE_OFFSET}_scalebar100um.svg",
        x_position=0,
        y_position=2.3,
        x_offset=0.08,
        y_offset=0.08,
    ),
    FigurePanel(
        letter="F",
        path=save_dir_2 / f"bf_outliers_{dataset_config_bf.name}_P{position}.svg",
        x_position=0,
        y_position=5.4,
        x_offset=0.1,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="G",
        path=save_dir_2 / f"gfp_outliers_{dataset_config_gfp.name}_P{position}.svg",
        x_position=3.45,
        y_position=5.4,
        x_offset=-0.1,
        y_offset=0.1,
    ),
]

output_path = (
    get_output_path("supp_fig_z_slice_outliers") / "supp_fig_z_slice_selection_outliers.svg"
)
build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT)

# %%
