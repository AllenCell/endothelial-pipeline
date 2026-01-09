import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

from endo_pipeline.configs.dataset_config import TimepointAnnotation
from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io.input import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
from endo_pipeline.manifests.dataframe_manifest_utils import get_dataframe_location_for_dataset


def get_position_on_slide(
    df: pd.DataFrame, x_col: str = "centroid_x_um", y_col: str = "centroid_y_um"
) -> pd.DataFrame:
    """
    Calculate cell positions on the slide based on centroid positions and image metadata.

    Parameters
    ----------
    df
        DataFrame containing 'centroid_x_um', 'centroid_y_um',
        'image_size_x', 'image_size_y', 'pixel_size_xy_in_um'

    Returns
    -------
    :
        DataFrame with additional 'x_um_on_slide' and 'y_um_on_slide' columns

    """
    # Offset x positions based on image size and pixel size
    # There is a 10% overlap between positions
    offset_x_per_pos = df["image_size_x"].iloc[0] * df["pixel_size_xy_in_um"].iloc[0]

    for pos_index, df_pos in df.groupby("position"):
        x_offset = offset_x_per_pos * pos_index
        df.loc[df_pos.index, f"{x_col}_on_slide"] = df_pos[x_col] + x_offset
        df.loc[df_pos.index, f"{y_col}_on_slide"] = df_pos[y_col]
    return df


def get_dataframe_for_spatial_correlation_analysis(dataset_name: str) -> pd.DataFrame:
    # load dataset config
    dataset_config = load_dataset_config(dataset_name)

    # Load the tables with cdh5 segmentation measurements
    live_seg_manifest = load_dataframe_manifest("live_merged_seg_features")
    live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
    live_seg_feats_df = load_dataframe(live_seg_location)

    # filter out rows based on track-based features
    live_seg_feats_df = live_seg_feats_df[live_seg_feats_df.is_included]

    # filter out rows based on automatic and manual timepoint annotations
    live_seg_feats_df["dataset"] = live_seg_feats_df["dataset_name"]
    live_seg_feats_df["frame_number"] = live_seg_feats_df["image_index"]
    annotations_to_filter_out = [
        TimepointAnnotation.AUTO_GFP_SCOPE_ERROR,
        TimepointAnnotation.GFP_SCOPE_ERROR,
    ]
    live_seg_feats_df = filter_dataframe_by_annotations(
        live_seg_feats_df, dataset_config, timepoint_annotations=annotations_to_filter_out
    )

    # calculate features that are sensitive to how the dataframe is filtered
    live_seg_feats_df = calculate_derived_data_dynamics_dependent(live_seg_feats_df)

    # the original orientation feature is in radians
    # and the y-axis is defined as 0 degrees
    # this converts the orientation angle range between 0-180 degrees
    # and calculates cos(2*theta) for nematic correlations
    live_seg_feats_df["orientation_x"] = wrap_to_pi(
        live_seg_feats_df["orientation"].to_numpy() + np.pi / 2
    )
    live_seg_feats_df["cos_2_orientation"] = np.cos(2 * (live_seg_feats_df["orientation_x"]))

    # calculate cell positions on the slide
    live_seg_feats_df = get_position_on_slide(live_seg_feats_df)

    return live_seg_feats_df


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_r_bins(
    x: np.ndarray,
    y: np.ndarray,
    bin_width: float = 1.0,
    r_min: float = 0.0,
    r_max: float | None = None,
) -> np.ndarray:
    """
    Create distance bins for pairwise correlation metrics.

    Parameters
    ----------
    x
        X coordinates
    y
        Y coordinates
    bin_width
        Width of distance bins
    r_min
        Minimum distance
    r_max
        Maximum distance. If None, set to half the minimum window side length

    Returns
    -------
    :
        Array of bin edges with length n_bins + 1
    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    Lx, Ly = xmax - xmin, ymax - ymin
    r_max_value = r_max if r_max is not None else 0.5 * min(Lx, Ly)
    bin_centers = np.arange(r_min, r_max_value + 0.5 * bin_width, bin_width)
    bin_edges = np.concatenate([bin_centers - 0.5 * bin_width, [bin_centers[-1] + 0.5 * bin_width]])
    return bin_edges


def exp_decay(x: np.ndarray, lam: float) -> np.ndarray:
    """Exponential decay function."""
    return np.exp(-x / lam)


def pairwise_correlation(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    r_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray, np.ndarray]:
    """
    Compute spatial autocorrelation function C(r).

    Calculate C(r) = <f_i · f_j> / <f²> for pairs separated by distance r.
    This measures similarity of feature values as a function of distance.
    For nematic features (like cos(2θ)), this gives the nematic correlation function.

    The normalization ensures C(0) = 1 when the first bin includes r=0.

    Parameters
    ----------
    x
        X coordinates of points
    y
        Y coordinates of points
    f
        Function values at each point (e.g., cos(2*orientation) for nematic correlations)
    r_bins
        Distance bin edges in same units as coordinates

    Returns
    -------
    :
        Tuple containing:
        - r_centers: bin center distances
        - C_r: correlation per bin (normalized so C(0) = 1)
        - xi_1_over_e: correlation length where C(r) = 1/e, or None if not found
        - counts: number of pairs per bin
        - fitted_c_r: fitted exponential decay values at r_centers
    """
    points = np.column_stack([x, y])
    pairwise_dists = squareform(pdist(points, metric="euclidean"))

    # Compute pairwise products f_i * f_j
    correlation_matrix = np.outer(f, f)

    # Get upper triangle indices (i < j) plus diagonal for all pairs
    triu_indices = np.triu_indices_from(pairwise_dists)
    flat_dists = pairwise_dists[triu_indices]
    flat_corr = correlation_matrix[triu_indices]

    # Bin the distances and correlations
    nbins = len(r_bins) - 1
    bin_ids = np.digitize(flat_dists, r_bins) - 1
    valid = (bin_ids >= 0) & (bin_ids < nbins)

    sum_val = np.bincount(bin_ids[valid], weights=flat_corr[valid], minlength=nbins)
    counts = np.bincount(bin_ids[valid], minlength=nbins).astype(int)

    C_r = np.full(nbins, np.nan)
    ok = counts > 0
    # Normalize by <f²> to ensure C(0) = 1
    mean_f_squared = np.mean(f**2)
    C_r[ok] = sum_val[ok] / (counts[ok] * mean_f_squared)

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Fit exponential decay to extract correlation length
    xi_1_over_e = None
    fitted_c_r = C_r.copy()
    if np.any(ok) and np.sum(ok) >= 2:
        try:
            popt, _ = curve_fit(
                exp_decay,
                r_centers[ok],
                C_r[ok],
                p0=[np.mean(r_centers[ok])],
                bounds=(0, np.inf),  # Ensure positive correlation length
            )
            xi_1_over_e = popt[0]
            fitted_c_r = exp_decay(r_centers, *popt)
        except Exception:
            xi_1_over_e = None

    return r_centers, C_r, xi_1_over_e, counts, fitted_c_r


def get_delaunay_triangulation(
    x: np.ndarray,
    y: np.ndarray,
) -> Delaunay:
    """
    Compute Delaunay triangulation for points.

    Parameters
    ----------
    x
        X coordinates of points
    y
        Y coordinates of points

    Returns
    -------
    :
        Delaunay triangulation object
    """
    points = np.column_stack([x, y])
    return Delaunay(points)


def calculate_topological_defects(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    wind_threshold: float = 0.35,
    merge_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Detect topological defects using Delaunay triangulation.

    Identify ±1/2 defects by computing circulation of doubled angles
    around triangle edges. Total circulation ≈ ±2π indicates a defect.

    Merges defects within merge_radius (set to mean edge length if None).

    Parameters
    ----------
    x
        X coordinates of points
    y
        Y coordinates of points
    theta
        Orientation angles in radians
    wind_threshold: float = 1.5 * np.pi,
        Minimum winding number magnitude to consider a defect
    merge_radius
        Radius to merge nearby defects. If None, set to mean edge length of triangulation.

    Returns
    -------
    :
        Tuple containing:
        - List of detected defects as tuples (x, y)
        - List of corresponding winding numbers
        - Mean nearest-defect distance
    """
    tri = get_delaunay_triangulation(x, y)

    phi = 2 * theta  # Convert orientation to nematic angle

    defect_positions = []
    defect_numbers = []

    for simplex in tri.simplices:
        i, j, k = simplex
        dphi_ij = wrap_to_pi(phi[j] - phi[i])
        dphi_jk = wrap_to_pi(phi[k] - phi[j])
        dphi_ki = wrap_to_pi(phi[i] - phi[k])

        winding_number = (dphi_ij + dphi_jk + dphi_ki) / (4 * np.pi)

        if np.abs(winding_number) >= wind_threshold:
            x_defect = (x[i] + x[j] + x[k]) / 3.0
            y_defect = (y[i] + y[j] + y[k]) / 3.0
            defect_positions.append((x_defect, y_defect))
            defect_numbers.append(winding_number)

    # Merge nearby defects
    if merge_radius is None and len(defect_positions) > 1:
        # Estimate merge radius as mean edge length
        edges = []
        for simplex in tri.simplices:
            i, j, k = simplex
            edges.extend(
                [
                    np.hypot(x[i] - x[j], y[i] - y[j]),
                    np.hypot(x[j] - x[k], y[j] - y[k]),
                    np.hypot(x[k] - x[i], y[k] - y[i]),
                ]
            )
        merge_radius = np.mean(edges).item()

    defect_assigned = np.zeros(len(defect_positions), dtype=bool)
    merged_defect_positions = []
    merged_defect_numbers = []
    for i, (pos_i, num_i) in enumerate(zip(defect_positions, defect_numbers, strict=False)):
        if defect_assigned[i]:
            continue
        x_i, y_i = pos_i
        merged_pos = np.array(pos_i)
        merged_num = num_i
        count = 1

        for j in range(i + 1, len(defect_positions)):
            if defect_assigned[j]:
                continue
            x_j, y_j = defect_positions[j]
            dist_ij = np.hypot(x_i - x_j, y_i - y_j)
            if dist_ij <= merge_radius:
                merged_pos += np.array(defect_positions[j])
                merged_num += defect_numbers[j]
                count += 1
                defect_assigned[j] = True

        merged_pos /= count
        merged_defect_positions.append((merged_pos[0], merged_pos[1]))
        merged_defect_numbers.append(merged_num)

    # Compute mean nearest-defect distance
    mean_nearest_defect_distance = None
    if len(merged_defect_positions) > 1:
        defect_points = np.array(merged_defect_positions)
        pairwise_defect_dists = pdist(defect_points, metric="euclidean")

        dist_matrix = squareform(pairwise_defect_dists)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_dists = np.min(dist_matrix, axis=1)
        mean_nearest_defect_distance = np.mean(nearest_dists).item()

    return (
        np.array(merged_defect_positions),
        np.array(merged_defect_numbers),
        mean_nearest_defect_distance,
    )
