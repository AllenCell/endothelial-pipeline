import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.util import manifest_io
from src.endo_pipeline.configs import DatasetConfig, ModelManifest, load_dataset_config
from src.endo_pipeline.io import load_dataframe_from_fms


def get_valid_subset(df: pd.DataFrame, dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Select timepoints from a dataframe annotated as valid
    if annotation is present, otherwise use all timepoints.

    Inputs:
    - df: pd.DataFrame, containing the metadata for the dataset name and timepoints
    - dataset_name: str, name of the dataset to get valid timepoints for

    Outputs:
    - df: pd.DataFrame, subset of the input dataframe containing only the valid timepoints
    """
    df["valid"] = False
    # check that the necessary datasets are present for fitting PCA
    valid_timepoints = load_dataset_config(dataset_name).valid_timepoints
    if valid_timepoints is None:
        if verbose:
            print(f"Using all timepoints from dataset {dataset_name} for analysis")
        df["valid"] = True
    else:
        if verbose:
            print(f"Valid timepoints for dataset {dataset_name}: ")
        tps = []
        for start, stop in zip(valid_timepoints.start, valid_timepoints.stop, strict=True):
            tps.extend(list(range(start, stop + 1)))
            if verbose:
                print(f"   - {start} to {stop}")
        valid_subset = df.frame_number.isin(tps)
        df["valid"] = valid_subset
    return df[df.valid]


def add_description_column(df: pd.DataFrame, ds_name: str, simple: bool = False) -> pd.DataFrame:
    """
    Add description column to DataFrame df.
    (Descriptions are currently based on the dataset name.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data for dataset ds_name
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column
    - ds_name: str, name of dataset to add description for
    - simple (optional): bool, whether to use simple description
        (e.g., "48hr_High")

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added description column
    """
    # get descriptions for each dataset name
    description = get_dataset_descriptions([ds_name], simple=simple)

    # add description column to DataFrame
    df["description"] = description[ds_name]  # add description to DataFrame

    return df


def get_dataset_descriptions(list_of_datasets: list[str], simple: bool = False) -> dict:
    """
    Get descriptive metadata for each dataset given in the list of datasets.

    Describes the experimental conditions for each dataset,
        e.g., "48_hours_at_30_dyncm2".

    Inputs:
    - list_of_datasets: list, list of dataset names to get descriptions for
        - Each string should match the appropriate dataset name in data_config.yaml
    - simple (optional): bool, whether to use simple description (e.g., "48hr_High")


    Outputs:
    - description_dic: dict, dictionary of dataset names and their descriptive metadata
    """

    # initialize dictionary to store descriptions
    description_dic = {}
    for name in list_of_datasets:
        data_config = load_dataset_config(name)  # get dataset info from data_config.yaml
        flow_config = data_config.flow  # get flow conditions for dataset
        num_flows = len(flow_config)  # number of flow conditions in dataset

        # get shear rate for each flow condition,
        # last element in each list in flow_config
        shear_rate = [int(flow_config[i][-1]) for i in range(num_flows)]
        if simple:  # if simple description, use qualitative description of shear stress level
            shear_rate_str = []
            for shear in shear_rate:
                if shear >= 20:
                    shear_rate_str.append("High")
                elif shear > 7:
                    shear_rate_str.append(f"Intermediate_{int(shear)}")
                elif shear > 0:
                    shear_rate_str.append("Low")
                else:
                    shear_rate_str.append("No")
        else:
            shear_rate_str = [
                f"{int(i)}_dyncm2" for i in shear_rate
            ]  # convert shear rates to strings

        time_str = [
            f"{int((flow_config[i][1]-flow_config[i][0])*5/60)}hr" for i in range(num_flows)
        ]  # get duration of each flow condition in hours
        description = "_".join(
            [time_str[i] + "_" + shear_rate_str[i] for i in range(num_flows)]
        )  # concatenate time and shear rate for each flow condition
        description_dic[name] = description  # add description to dictionary

    return description_dic


def add_crop_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add crop index column to DataFrame df. (Crops are currently identified by
        their starting position in x and y.).

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata
        columns for start_x, start_y, and FOV_ID
        - IMPORTANT: DataFrame must be restricted to one dataset only,
            as identified by the dataset_name column

    Outputs:
    - df: pd.DataFrame, DataFrame of feature data for one
        dataset with added crop index column
    """
    assert "start_x" in df.columns, "Data must have a column for start_x"
    assert "start_y" in df.columns, "Data must have a column for start_y"
    assert "position" in df.columns, "Data must have a column for position"

    # get list of unique starting positions and FOV_IDs
    # (position column in DiffAE manifest)
    start_x = df[df["frame_number"] == df["frame_number"].min()]["start_x"].values.tolist()
    start_y = df[df["frame_number"] == df["frame_number"].min()]["start_y"].values.tolist()
    position = df[df["frame_number"] == df["frame_number"].min()]["position"].values.tolist()
    tup_list = list(zip(start_x, start_y, position, strict=True))

    # function to convert starting position and FOV_ID to crop index
    def _pos_to_index(x: float, y: float, position: str) -> int:
        return tup_list.index((x, y, position))

    # apply function to DataFrame to get crop index
    df["crop_index"] = df.apply(
        lambda x: _pos_to_index(x["start_x"], x["start_y"], x["position"]), axis=1
    )

    return df


def project_manifest_to_pcs(
    df: pd.DataFrame,
    pca: Pipeline,
    overwrite_feature_columns: bool = True,
    feat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Project feature data for crops from one dataset onto principal
    component axes of fit PCA model.

    Inputs:
    - df: pd.DataFrame, DataFrame of feature data with metadata columns
        for dataset_name, T, FOV_ID, start_x, start_y
    - pca: Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - can include any preprocessing steps before PCA, e.g., scaling
    - ds_name: str, name of dataset to project feature data for
        - This string must match the dataset name in the dataset_name column of df, same
           as the name of the dataset in data_config.yaml
    - feature_cols: list, custom list of feature columns to project onto PCA axes

    Outputs:
    - df_: pd.DataFrame, DataFrame of feature data for crops from
        dataset ds_name projected onto PCA axes
    """
    # feature columns to project onto PCA axes,
    # currently all columns except metadata columns
    # this is assuming that there are 8 feature columns,
    # will need to change if this is not the case
    if feat_cols is None:
        feat_cols = manifest_io.get_feature_cols(df)

    df_ = df.copy()  # make copy of DataFrame to avoid modifying original DataFrame

    # project feature data onto PCA axes, replace feature columns
    # with features projected onto PCA axes
    if overwrite_feature_columns:
        df_.loc[:, feat_cols] = pca.transform(df_[feat_cols].values)
    else:
        pc_cols = [f"pc{pc+1}" for pc in range(len(feat_cols))]
        df_.loc[:, pc_cols] = pca.transform(df_[feat_cols].values)

    return df_


def get_manifest_for_dynamics_workflows(
    model_manifest: ModelManifest, pca: Pipeline | None = None
) -> pd.DataFrame:
    """
    Load DiffAE manifest data projected onto given PC axes for downstream analysis
    in the stochastic dynamics workflow. Adds crop index column to DataFrame,
    and projects feature data onto PC axes.

    Inputs:
    - ds_name: str, name of dataset to load manifest data for
        - This string must match the dataset name in the dataset_name column of df, same
           as the name of the dataset in data_config.yaml
    - pca: Pipeline or None
        - if Pipeline, PCA model fit to feature data (using sklearn.pipeline.Pipeline)
        - if None, do not project feature data onto PCA axes

    Outputs:
    - pd.DataFrame of feature data for crops
        from input model_manifest
        - projected onto PC axes if pca is not None
        - restricted to stationary frames if
            stationary_frames is not None
    """
    # load manifest data for dataset ds_name
    # and filter to only valid timepoints
    df = load_dataframe_from_fms(model_manifest.fmsid)
    df_valid = get_valid_subset(df, model_manifest.dataset_name, verbose=False)

    # add crop index column
    df_with_crop = add_crop_index(df_valid)

    if pca is None:
        # do not project feature data onto PCA axes
        return df_with_crop

    else:
        # project feature data onto PC axes
        return project_manifest_to_pcs(df_with_crop, pca)


def df_to_array(df_: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """
    Convert DataFrame of features corresponding to one dataset to array
    of shape num_crops x num_timepoints x num_features.

    Inputs:
    - df_: pd.DataFrame, DataFrame of feature data for one dataset
        - DataFrame should have metadata columns for crop_index and T

    Outputs:
    - feats: np.ndarray, array of feature data for all crops
        at all timepoints in one dataset
        - shape is num_crops x num_timepoints x num_features
    """
    assert "crop_index" in df_.columns, "DataFrame must have a column for crop_index"
    assert "frame_number" in df_.columns, "DataFrame must have a column for frame_number"

    num_time = df_["frame_number"].nunique()  # number of timepoints in the movie
    num_crop = df_["crop_index"].nunique()  # number of crops made at each timepoint

    # get array of num crops x num timepoints x num PCs
    feats = np.array(
        [
            df_[df_["crop_index"] == ii].sort_values(by="frame_number")[feat_cols].values
            for ii in range(num_crop)
        ]
    )

    # check that array shape is correct
    assert feats.shape == (num_crop, num_time, len(feat_cols))

    return feats


def get_timepoints_for_plotting_pcs(
    list_of_datasets: list[DatasetConfig],
    restrict_no_flow: bool = True,
    no_flow_name: str = "20241217_20X",
) -> dict:
    """
    Get timepoints for plotting scatter plot in PC
    space of data used to fit PCA.

    Used to remove later block of timepoints from the
    20241217_20X no flow dataset for generating "simplified"
    scatter plots for the 2025 SAC presentation.
    """
    # initialize dictionary to store timepoints for each dataset
    timepoints_to_use = {}

    for dataset in list_of_datasets:
        # get range of valid timepoints for each dataset
        # loaded from dataset config
        timepoint_dict = dataset.valid_timepoints

        # if no valid timepoints are specified, use all timepoints
        if timepoint_dict is None:
            timepoints_list = [[0, dataset.flow[0][1]]]

        # otherwise, get the start and stop timepoints
        else:
            starts = timepoint_dict.start
            stops = timepoint_dict.stop
            timepoints_list = []
            for start, stop in zip(starts, stops, strict=True):
                # hard coded because this is the no-flow dataset that
                # we are using for fitting the PCs, and specifically
                # the one with the two sets of timepoints
                # if this changes, we can updated this to not be
                # hardcoded (i.e., check if shear stress is 0 in config)
                if dataset.name == no_flow_name and restrict_no_flow:
                    # restrict to only first set of no flow timepoints
                    if start == 0:
                        timepoints_list.append([start, stop])
                    else:
                        continue
                else:
                    timepoints_list.append([start, stop])
        timepoints_to_use[dataset.name] = timepoints_list
    return timepoints_to_use
