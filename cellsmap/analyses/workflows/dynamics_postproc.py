# %%
import numpy as np
import pandas as pd
import pysindy as ps
import numdifftools as nd

import cellsmap.analyses.utils.gen_potential as gp
import cellsmap.analyses.utils.regression as eareg
from cellsmap.analyses.utils import manifest_viz, manifest_io, manifest_pca

# import config parameters
from cellsmap.analyses.configs.dynamics_postproc_config import savedir, PCs, ds_to_skip, ndim, Nbins

# %%
# make save directory for workflow outputs
manifest_io.make_savedir(savedir)
# %%
# load data
df = manifest_io.load_manifest_to_df()
# fit PCA to data
df, pca = manifest_pca.get_pca(df, num_pcs=8)
# add outliers to dataframe
df = manifest_pca.get_outliers(df)
# filepath for this dataset in manifest includes barcode, so we need to change the group name to match data config
# something that should be fixed in the manifest in the future
df.loc[df.group.str.contains('20250224'),'group'] = '20250224_20X'
# get list of datasets by 'group' identifier
list_of_datasets = manifest_io.get_list_of_datasets(df,'group',verbose=True)

# %%
# plot explained variance ratio of PCA components
fig, ax = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)
manifest_viz.save_plot(fig,filename=savedir+'figs/explained_variance_ratio',format='.png',dpi=500)
# %%
# plot top 3 principal components of feature data vs. frame number
# should write a function to do this for all datasets in the manifest via data_config
title_dict = manifest_io.get_descriptive_metadata(list_of_datasets)

fig, axs = manifest_viz.plot_top_3_PCs_alldata(df,pca['pca'],list_of_datasets, title_dict)
manifest_viz.save_plot(fig,filename=savedir+'figs/top_3_PCs',format='.png',dpi=500)

# %%
