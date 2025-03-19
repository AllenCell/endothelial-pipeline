from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import seaborn as sns
from cellsmap.util import dataset_io, cdh5_preprocessing as preproc
from cellsmap.features import cdh5_seg_density_map as cellsden

# silence the max number of plots warning
plt.rcParams.update({'figure.max_open_warning': 0})

def evaluate_density_against_number_of_nuclei(dataset_name, T, bbox_radius=256, nsamples=1000):
    np.random.seed(666)
    # The density maps from thresholds and segmentations are comparable form my test on dataset 20240305_T01_001
    # dmap = cellsden.get_density_map_from_segmentations(dataset_name, T, density_map_sigma=10)
    dmap = cellsden.get_density_map_from_thresholds(dataset_name, T, density_map_sigma=40)
    nuc_seg = dataset_io.load_dataset(dataset_name, channels=["Nuc_Seg"], time_start=T, time_end=T, level=0).squeeze()
    region_seg = np.array(*preproc.get_cdh5_classic_segmentation(dataset_name, T, channels=['segmentations_merged'])).squeeze()
    nuc_seg = nuc_seg.compute().astype(int)
    df = []
    r = bbox_radius
    rs = int(r*(0.5**2))
    for sample in range(nsamples):
        x = np.random.randint(r, nuc_seg.shape[-1]-r)
        y = np.random.randint(r, nuc_seg.shape[-2]-r)
        crop_region = slice(y-r, y+r), slice(x-r, x+r)

        xs, ys = int(x*(0.5**2)), int(y*(0.5**2))
        crop_region_small = slice(ys-rs, ys+rs), slice(xs-rs, xs+rs)

        nuc_crop = nuc_seg[crop_region]
        # loading the nuclear segmentation at a less-than-native resolution results in
        # blurring of the segmentation labels, therefore we must re-label them if loaded
        # like that (not an issue at native resolution though)
        n = len(regionprops(label(nuc_crop.astype(bool))))
        n_cdh5_regions = len(regionprops(region_seg[crop_region]))
        dens = dmap[crop_region_small].mean()
        dens_std = dmap[crop_region_small].std()
        df.append({"density": dens, "density_std":dens_std, "n_nuclei": n, "n_CDH5_regions": n_cdh5_regions})
    df = pd.DataFrame(df)

    pearson = np.corrcoef(df.density, df.n_nuclei)[0, 1]
    pearson2 = np.corrcoef(df.density, df.n_CDH5_regions)[0, 1]
    df['dataset'] = dataset_name
    df['T'] = T
    df['pearson_nuclei'] = pearson
    df['pearson_cdh5_regions'] = pearson2

    fig, axs = plt.subplots(figsize=(10,5), ncols=2, nrows=1)
    axs[0].scatter(df.n_nuclei, df.density, s=5)
    axs[0].set_xlabel("number of nuclei")
    axs[0].set_ylabel("density from thresholds")
    axs[0].set_title(f"Dataset: {dataset_name} \ntimepoint={T}, Pearson={pearson:.2f}")

    axs[1].scatter(df.n_nuclei, df.density, s=5)
    axs[1].set_xlabel("number of CDH5 regions")
    axs[1].set_ylabel("density from thresholds")
    axs[1].set_title(f"Dataset: {dataset_name} \ntimepoint={T}, Pearson={pearson:.2f}")
    plt.tight_layout()
    # plt.show()
    return df, pearson, fig

def save_results(out_dir, t, df, pearson, fig):
    Path.mkdir(out_dir / 'plots', exist_ok=True, parents=True)
    Path.mkdir(out_dir / 'tables', exist_ok=True, parents=True)
    fig.savefig(out_dir / f'plots/T{t}_pearson={pearson:.2f}.png')
    df.to_csv(out_dir / f'tables/T{t}_results.csv')


dataset_name_list = ['20240305_T01_001',]

for dataset_name in dataset_name_list:
    print(dataset_name)
    T_range = range(0, dataset_io.get_dataset_duration_in_frames(dataset_name), 6)

    prj_dir = Path('../').resolve()
    out_dir = prj_dir / 'results/cdh5_seg_density_map_correlations' / dataset_name

    for t in T_range:
        print(f'T={t}')
        df, pearson, fig = evaluate_density_against_number_of_nuclei(dataset_name, t, bbox_radius=256, nsamples=1000)
        save_results(out_dir, t, df, pearson, fig)

    table_paths = [fp for fp in out_dir.glob('tables/*.csv')]
    master_table = pd.concat([pd.read_csv(fp) for fp in table_paths])

    fig, ax = plt.subplots(figsize=(8,6), ncols=1, nrows=1)
    sns.lineplot(x='T', y='pearson_nuclei', data=master_table, ls='-', marker='o', ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(x='T', y='pearson_cdh5_regions', data=master_table, ls='-', marker='o', c='tab:orange', ax=ax2)
    ax.set_ylim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax.axhline(0, color='black', linestyle='--')
    [ax.axhline(y, color='grey', linestyle=':') for y in (-0.5, 0.5)]
    ax.yaxis.set_tick_params(labelcolor='tab:blue')
    ax2.yaxis.set_tick_params(labelcolor='tab:orange')
    ax.set_ylabel('Pearson correlation coefficient between threshold \n density map and number of nuclei')
    ax2.set_ylabel('Pearson correlation coefficient between threshold \n density map and number of CDH5 regions', rotation=-90, verticalalignment='bottom')
    ax.set_adjustable('box')
    plt.tight_layout()
    fig.savefig(out_dir / 'plots/T_vs_pearson.png')
