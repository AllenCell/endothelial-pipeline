from cellsmap.vis.timelapse_feature_explorer.backdrop_images import generate_backdrops, add_backdrop_fname_to_manifest
import ast

def update_manifest_for_tfe(df, dataset, position, output_dir):
    """
    Update the manifest DataFrame for TFE by adding necessary columns.
    """
    df["centroid"] = df["centroid"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["centroid_x"] = df["centroid"].apply(lambda x: x[1])
    df["centroid_y"] = df["centroid"].apply(lambda x: x[0])
    df["dataset"] = dataset
    df["position"] = position
    df["object_id"] = df.groupby(["position", "image_index", "label"]).ngroup() + 1 # plus one so object id is not 0, background
    df["seg_image"] = (
        ""
        # + df["dataset"]+ "/P" + df["position"].astype(str) + "/"
        + df["dataset"]
        + "_P" + df["position"].astype(str)
        + "_T" + df["T"].astype(str)
        + ".ome.tiff"
    )
    df = add_backdrop_fname_to_manifest(df, dataset, position, ["bf_slice", "bf_std_dev", "gfp_max_proj"], output_dir= output_dir / "backdrops")
    df["tid"] = df["track_id"]
    df.drop(columns=["centroid", "T", "reference_index", "matched_query_label", "optimized_metric_value"], inplace=True)  # Drop the original centroid column as it's no longer needed
    return df