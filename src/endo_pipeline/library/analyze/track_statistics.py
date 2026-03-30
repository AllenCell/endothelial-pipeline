import numpy as np
import pandas as pd
from scipy.stats import circmean, circvar

from endo_pipeline.settings.column_names import ColumnName as Column


def compute_track_statistics(
    dataframe: pd.DataFrame,
    column_names: list[str],
    trajectory_id_col: str = Column.CROP_INDEX,
    polar_angle_range: tuple[float, float] = (0, np.pi),
    average_col_suffix: str = "_average",
    variance_col_suffix: str = "_variance",
) -> pd.DataFrame:
    """
    Compute average and variance over specified columns for each trajectory.

    Parameters
    ----------
    dataframe
        The input dataframe containing the trajectory data.
    column_names
        The list of column names for which to compute statistics.
    trajectory_id_col
        The name of the column containing trajectory IDs.
    polar_angle_range : tuple[float, float], optional
        The range of polar angles for circular statistics, by default (0, np.pi)
    """
    average_col_names = [f"{col}{average_col_suffix}" for col in column_names]
    variance_col_names = [f"{col}{variance_col_suffix}" for col in column_names]
    stats_df = pd.DataFrame(columns=[trajectory_id_col, *average_col_names, *variance_col_names])
    for traj_index, df_traj in dataframe.groupby(trajectory_id_col):
        for column_name in column_names:
            if column_name == Column.DiffAEData.POLAR_ANGLE:
                # take circular mean for polar angle to account for periodicity
                stats_df.loc[traj_index, f"{column_name}{average_col_suffix}"] = circmean(
                    df_traj[column_name],
                    high=polar_angle_range[1],
                    low=polar_angle_range[0],
                )
                stats_df.loc[traj_index, f"{column_name}{variance_col_suffix}"] = circvar(
                    df_traj[column_name],
                    high=polar_angle_range[1],
                    low=polar_angle_range[0],
                )
            else:
                stats_df.loc[traj_index, f"{column_name}{average_col_suffix}"] = np.nanmean(
                    df_traj[column_name]
                )
                stats_df.loc[traj_index, f"{column_name}{variance_col_suffix}"] = np.nanvar(
                    df_traj[column_name]
                )

    return stats_df
