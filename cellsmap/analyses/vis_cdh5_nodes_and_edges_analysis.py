from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.projections.polar import PolarAxes
import seaborn as sns

def set_max_plots(max_plots: int=plt.rcParams['figure.max_open_warning']):
    """
    Changes the number of plots that can be opened before a warning is raised.
    If max_plots=0 then the warning will be silenced (i.e. no limit).
    Default is determined by plt.rcParams['figure.max_open_warning'] (usually 20).

    Parameters
    ----------
    max_plots: int
        Number of plots that can be opened before a warning is raised.
    """
    plt.rcParams.update({'figure.max_open_warning': max_plots})

def generate_alignment_summary_plots(df_ang_dist_summary: pd.DataFrame, out_dir: Path, SHOW_PLOTS: bool, SAVE_OUTPUT: bool):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='node_to_node_distance_mean', y='angle_relative_to_horizontal_in_deg_mean',
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=9, ax=ax)
    ax.plot(df_ang_dist_summary['node_to_node_distance_mean'], df_ang_dist_summary['angle_relative_to_horizontal_in_deg_mean'],
                    lw=1, c='lightgrey', zorder=1)
    flow_change = df_ang_dist_summary.query('`Time (hours)` == 24')
    death_event = df_ang_dist_summary.query('`T` == 107')
    sns.scatterplot(data=flow_change, x='node_to_node_distance_mean', y='angle_relative_to_horizontal_in_deg_mean',
                    facecolor='none', edgecolor='k', marker='o', zorder=10, legend=False, ax=ax)
    sns.scatterplot(data=death_event, x='node_to_node_distance_mean', y='angle_relative_to_horizontal_in_deg_mean',
                    facecolor='none', edgecolor='r', marker='o', zorder=10, legend=False, ax=ax)
    ax.set_ylabel('Mean angle relative to horizontal (degrees)')
    ax.set_xlabel('Mean node-node distance (px)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'angles_vs_dists_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None

    # NOTE: Why... is the mean node-node distance increasing under high flow with essentially
    #   no change in the mean angle...? Is this actually seen in the real data or is it an
    #   artifact of the way nodes and edges are determined?
    #       - according to Becky the cells are actually elongating


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='node_to_node_distance_std', y='angle_relative_to_horizontal_in_deg_std',
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=9, ax=ax)
    ax.plot(df_ang_dist_summary['node_to_node_distance_std'], df_ang_dist_summary['angle_relative_to_horizontal_in_deg_std'],
                    lw=1, c='lightgrey', zorder=1)
    flow_change = df_ang_dist_summary.query('`Time (hours)` == 24')
    death_event = df_ang_dist_summary.query('`T` == 107')
    sns.scatterplot(data=flow_change, x='node_to_node_distance_std', y='angle_relative_to_horizontal_in_deg_std',
                    facecolor='none', edgecolor='k', marker='o', zorder=10, legend=False, ax=ax)
    sns.scatterplot(data=death_event, x='node_to_node_distance_std', y='angle_relative_to_horizontal_in_deg_std',
                    facecolor='none', edgecolor='r', marker='o', zorder=10, legend=False, ax=ax)
    ax.set_ylabel('StDev of angle relative to horizontal (degrees)')
    ax.set_xlabel('StDev of node-node distance (px)')
    ax.legend(title='Time (hours)', ncols=4)
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'angles_vs_dists_std.pdf')
    plt.close('all') if not SHOW_PLOTS else None


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_ylabel('Mean node-node distance (px)')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_dists_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                    marker='.', c='k', zorder=10, ax=ax)
    ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_xlim(5, 10)
    ax.set_ylabel('Mean node-node distance (px)')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_dists_means_closeup.pdf')
    plt.close('all') if not SHOW_PLOTS else None


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax2 = ax.twinx()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='angle_relative_to_horizontal_in_deg_mean',
                    marker='.', c='tab:orange', zorder=10, ax=ax2)
    ax2.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['angle_relative_to_horizontal_in_deg_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.set_xlabel('Time (hours)')
    ax.tick_params('x', width=2)
    ax.xaxis.minorticks_on()
    ax.set_ylabel('Mean node-node distance (px)')
    ax.tick_params('y', color='tab:blue', width=2)
    ax2.set_ylabel('Mean angle relative to horizontal (degrees)', rotation=270, verticalalignment='bottom')
    ax2.tick_params('y', color='tab:orange', width=2)
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_dists_means_and_angles.pdf')
    plt.close('all') if not SHOW_PLOTS else None


    # plot the intensity and length data too
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='edge_length (px)_mean',
                    marker='.', c='k', legend=True, zorder=10, ax=ax)
    ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['edge_length (px)_mean'],
            lw=1, c='k', alpha=0.5, zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax2 = ax.twinx()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='node_to_node_distance_mean',
                    marker='.', c='tab:orange', legend=True, zorder=10, ax=ax2)
    ax2.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['node_to_node_distance_mean'],
             lw=1, c='tab:orange', alpha=0.5, zorder=1)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Mean edge length (px)')
    ax2.set_ylabel('Mean node-to-node distance (px)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_node-node-dists_and_edge-lengths.pdf')
    plt.close('all') if not SHOW_PLOTS else None
    ## NOTE SOMETHING IS WEIRD ABOUT THE NORMALIZED NODE_NODE DISTANCE
    ##      NOTICE HOW IT DROPS BELOW 1 (THIS SHOULD NOT BE POSSIBLE
    ##      SINCE THE EDGE LENGTH SHOULD NEVER BE SHORTER THAN A STRAIGHT
    ##      LINE BETWEEN THE 2 NODES.)
    ##          NOTE: IDEA: I BET IT'S BECAUSE EDGE LENGTH IS IN PIXELS
    ##                      AND DOESN'T ACCOUNT FOR DIAGONALS CORRECTLY
    ##          NOTE: THIS WAS PARTIALLY CORRECT, BUT THE edge_length (px)
    ##                      MEASURE IS ALSO MISSING THE EDGE-TO-NODE DISTANCES


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='Time (hours)', y='edge_fluorescence_mean (a.u.)_mean',
                    marker='.', c='k', legend=True, zorder=10, ax=ax)
    ax.plot(df_ang_dist_summary['Time (hours)'], df_ang_dist_summary['edge_fluorescence_mean (a.u.)_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Mean edge fluorescence (a.u.)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_edge_fluors_mean.pdf')
    plt.close('all') if not SHOW_PLOTS else None


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='node_to_node_distance_mean', y='edge_fluorescence_mean (a.u.)_mean',
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=9, ax=ax)
    ax.plot(df_ang_dist_summary['node_to_node_distance_mean'], df_ang_dist_summary['edge_fluorescence_mean (a.u.)_mean'],
                    lw=1, c='lightgrey', zorder=1)
    flow_change = df_ang_dist_summary.query('`Time (hours)` == 24')
    death_event = df_ang_dist_summary.query('`T` == 107')
    sns.scatterplot(data=flow_change, x='node_to_node_distance_mean', y='edge_fluorescence_mean (a.u.)_mean',
                    facecolor='none', edgecolor='k', marker='o', zorder=10, legend=False, ax=ax)
    sns.scatterplot(data=death_event, x='node_to_node_distance_mean', y='edge_fluorescence_mean (a.u.)_mean',
                    facecolor='none', edgecolor='r', marker='o', zorder=10, legend=False, ax=ax)
    ax.set_ylabel('Mean of edge fluorescence (a.u.)')
    ax.set_xlabel('Mean of node-node distance (px)')
    ax.legend(title='Time (hours)', ncols=4)
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'dists_vs_fluors_mean.pdf')
    plt.close('all') if not SHOW_PLOTS else None


    fig, ax = plt.subplots()
    sns.scatterplot(data=df_ang_dist_summary, x='angle_relative_to_horizontal_in_deg_mean', y='edge_fluorescence_mean (a.u.)_mean',
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=9, ax=ax)
    ax.plot(df_ang_dist_summary['angle_relative_to_horizontal_in_deg_mean'], df_ang_dist_summary['edge_fluorescence_mean (a.u.)_mean'],
                    lw=1, c='lightgrey', zorder=1)
    flow_change = df_ang_dist_summary.query('`Time (hours)` == 24')
    death_event = df_ang_dist_summary.query('`T` == 107')
    sns.scatterplot(data=flow_change, x='angle_relative_to_horizontal_in_deg_mean', y='edge_fluorescence_mean (a.u.)_mean',
                    facecolor='none', edgecolor='k', marker='o', zorder=10, legend=False, ax=ax)
    sns.scatterplot(data=death_event, x='angle_relative_to_horizontal_in_deg_mean', y='edge_fluorescence_mean (a.u.)_mean',
                    facecolor='none', edgecolor='r', marker='o', zorder=10, legend=False, ax=ax)
    ax.set_ylabel('Mean of edge fluorescence (a.u.)')
    ax.set_xlabel('Mean angle relative to horizontal (degrees)')
    ax.legend(title='Time (hours)', ncols=4)
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'angles_vs_fluors_mean.pdf')
    plt.close('all') if not SHOW_PLOTS else None



def generate_alignment_plots(out_path, filename_stem, timepoint, angles, distances, dist_min, dist_max, SHOW_PLOTS, SAVE_OUTPUT):

    print(filename_stem)

    angle_hists_path = out_path / 'angle_hists'
    Path.mkdir(angle_hists_path, parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection': 'polar'})
    ## bins are set up to be every 5 degrees
    ax.hist(angles, bins=18, facecolor='k')
    ax.set_xlim(0, np.pi/2)
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Count', horizontalalignment='center')
    ax.set_title(f'{timepoint:.3f} hours', loc='right')
    plt.tight_layout()
    if SAVE_OUTPUT:
        fig.savefig(angle_hists_path / (filename_stem + '_angles.tif'))

    color = 'tab:red' if timepoint <= 24.0 else 'tab:blue'
    angles_vs_dists_path = out_path / 'angles_vs_dists_polar'
    Path.mkdir(angles_vs_dists_path, parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection': 'polar'})
    ax.scatter(angles, distances, marker='.', c='k', alpha=0.3)
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(dist_min, dist_max)
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s='Node-node distance', horizontalalignment='center')
    ax.set_title(f'{timepoint:.3f} hours', loc='right', c=color)
    plt.tight_layout()
    if SAVE_OUTPUT:
        fig.savefig(angles_vs_dists_path / (filename_stem + '_dists_vs_angles_polar.tif'))

    angles_vs_dists_path = out_path / 'angles_vs_dists'
    Path.mkdir(angles_vs_dists_path, parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(angles, distances, marker='.', c='k', alpha=0.3)
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(dist_min, dist_max)
    ax.set_xlabel('Mean angle relative to horizontal (degrees)')
    ax.set_ylabel('Mean node-node distance (px)')
    ax.set_title(f'{timepoint:.3f} hours', loc='right')
    plt.tight_layout()
    if SAVE_OUTPUT:
        fig.savefig(angles_vs_dists_path / (filename_stem + '_dists_vs_angles.tif'))
    plt.close('all') if not SHOW_PLOTS else None


def compare_metrics_temporal_colorcode(df: pd.DataFrame, 
                                       x: str | None = None, 
                                       y: str | None = None, 
                                       semilog=False, 
                                       out_path: Path | None = None, 
                                       filename_stem: str | None = None, 
                                       SHOW_PLOTS=False, 
                                       SAVE_OUTPUT=False):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(data=df, x=x, y=y,
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), alpha=0.3,
                    linewidth=0, size=1, legend=False, zorder=10, ax=ax)
    ax.plot((0,1e4), (0,1e4), c='lightgrey', ls='--', lw=0.5, zorder=10)
    if semilog:
        ax.semilogx()
        ax.semilogy()
    axismin, axismax = min(df[x].min(), df[y].min()), max(df[x].max(), df[y].max())
    ax.set_xlim(axismin, axismax)
    ax.set_ylim(axismin, axismax)
    if SAVE_OUTPUT:
        assert out_path is not None and filename_stem is not None, 'Must provide out_path and filename_stem to save output.'
        fig.savefig(out_path / (filename_stem + f'_{x}_vs_{y}.pdf'))
    plt.close('all') if not SHOW_PLOTS else None

    # fig, ax = plt.subplots(figsize=(6,6))
    # sns.scatterplot(data=df, x=x, y=y,
    #                 marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), alpha=0.3,
    #                 linewidth=0, size=1, legend=False, zorder=10, ax=ax)
    # if SAVE_OUTPUT:
    #     fig.savefig(out_path / (filename_stem + f'_{x}_vs_{y}.pdf'))
    # plt.close('all') if not SHOW_PLOTS else None


def compare_metrics_temporal_colorcode_polar(df: pd.DataFrame, 
                                             x: str | None=None, 
                                             y: str | None=None, 
                                             out_path: Path | None=None, 
                                             filename_stem: str | None=None, 
                                             SHOW_PLOTS=False, 
                                             SAVE_OUTPUT=False):
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})
    assert isinstance(ax, PolarAxes), 'Axes must be polar for this plot.'
    sns.scatterplot(data=df, x=x, y=y,
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), alpha=0.3,
                    linewidth=0, size=1, legend=False, zorder=10, ax=ax)
    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(0,600)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(x=np.deg2rad(-12), y=ax.get_rmax()/2, s=ax.get_xlabel(), horizontalalignment='center')
    if SAVE_OUTPUT:
        assert out_path is not None and filename_stem is not None, 'Must provide out_path and filename_stem to save output.'
        fig.savefig(out_path / (filename_stem + f'_{x}_vs_{y}_polar.pdf'))
    plt.close('all') if not SHOW_PLOTS else None


def generate_segprop_summary_plots(df_segprops_summary: pd.DataFrame, out_dir: Path, SHOW_PLOTS, SAVE_OUTPUT):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segprops_summary, x='Time (hours)', y='cell_orientation_relative_to_horizontal_in_deg_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_segprops_summary['Time (hours)'], df_segprops_summary['cell_orientation_relative_to_horizontal_in_deg_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_ylabel('Mean cell orientation (degrees)')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_cell_orientation_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segprops_summary, x='Time (hours)', y='cell_area (px**2)_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_segprops_summary['Time (hours)'], df_segprops_summary['cell_area (px**2)_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_ylabel('Mean cell area (px$^2$)')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_cell_area_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segprops_summary, x='Time (hours)', y='cell_perimeter (px)_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_segprops_summary['Time (hours)'], df_segprops_summary['cell_perimeter (px)_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_ylabel('Mean cell perimeter (px)')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_cell_perimeter_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None

    # the eccentricity looks really good!
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segprops_summary, x='Time (hours)', y='cell_eccentricity_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_segprops_summary['Time (hours)'], df_segprops_summary['cell_eccentricity_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_ylabel('Mean cell eccentricity')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_cell_eccentricity_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None
    # the average fluorescence plot really reminds me of some of the principal components that
    # Erin has plotted
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segprops_summary, x='Time (hours)', y='cell_fluorescence_mean (a.u.)_mean',
                    marker='.', c='tab:blue', zorder=10, ax=ax)
    ax.plot(df_segprops_summary['Time (hours)'], df_segprops_summary['cell_fluorescence_mean (a.u.)_mean'],
                    lw=1, c='lightgrey', zorder=1)
    ax.axvline(24, c='k', ls='--')
    ax.axvline(107/12, c='r', ls='--')
    ax.set_ylabel('Mean of mean cell fluorescences')
    ax.set_xlabel('Time (hours)')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'time_vs_mean_cell_fluor_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segprops_summary, x='cell_eccentricity_mean', y='cell_orientation_relative_to_horizontal_in_deg',
                    marker='.', hue='Time (hours)', palette=plt.get_cmap('turbo'), legend=True, zorder=9, ax=ax)
    ax.plot(df_segprops_summary['cell_eccentricity_mean'], df_segprops_summary['cell_orientation_relative_to_horizontal_in_deg'],
                    lw=1, c='lightgrey', zorder=1)
    flow_change = df_segprops_summary.query('`Time (hours)` == 24')
    death_event = df_segprops_summary.query('`T` == 107')
    sns.scatterplot(data=flow_change, x='cell_eccentricity_mean', y='cell_orientation_relative_to_horizontal_in_deg',
                    facecolor='none', edgecolor='k', marker='o', zorder=10, legend=False, ax=ax)
    sns.scatterplot(data=death_event, x='cell_eccentricity_mean', y='cell_orientation_relative_to_horizontal_in_deg',
                    facecolor='none', edgecolor='r', marker='o', zorder=10, legend=False, ax=ax)
    ax.set_ylabel('Mean angle relative to horizontal (degrees)')
    ax.set_xlabel('Mean eccentricity')
    if SAVE_OUTPUT:
        fig.savefig(out_dir / 'orientations_vs_eccentricities_means.pdf')
    plt.close('all') if not SHOW_PLOTS else None
